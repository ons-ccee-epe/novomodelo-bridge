"""Tests for :mod:`novomodelo_bridge.cli.conversion_manifest`."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

import novomodelo_bridge
from novomodelo_bridge.cli.conversion_manifest import ConversionManifest
from novomodelo_bridge.cli.conversion_manifest import git_sha as conversion_git_sha
from novomodelo_bridge.core.diagnostics import Diagnostic, Severity
from novomodelo_bridge.core.provenance import hash_input_files, summarize_diagnostics
from tests.conftest import make_nw_files


def _in_git_checkout() -> bool:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return False
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def _diagnostic(code: str, severity: Severity) -> Diagnostic:
    """Build a minimal :class:`Diagnostic` for the given *severity*."""
    return Diagnostic(
        code=code,
        severity=severity,
        category="Test",
        title=code,
        summary=f"{code} summary",
    )


def test_hash_input_files_hashes_present_files(tmp_path: Path) -> None:
    dger_bytes = b"DGER-content-\x00\x01\x02"
    hidr_bytes = b"binary-\xff\xfe-payload"
    dger_path = tmp_path / "dger.dat"
    hidr_path = tmp_path / "hidr.dat"
    dger_path.write_bytes(dger_bytes)
    hidr_path.write_bytes(hidr_bytes)

    files = make_nw_files(tmp_path, dger=dger_path, hidr=hidr_path)

    entries = hash_input_files(files)

    by_field = {entry["field"]: entry for entry in entries}
    assert by_field["dger"]["sha256"] == hashlib.sha256(dger_bytes).hexdigest()
    assert by_field["dger"]["size_bytes"] == len(dger_bytes)
    assert by_field["dger"]["path"] == str(dger_path)
    assert by_field["hidr"]["sha256"] == hashlib.sha256(hidr_bytes).hexdigest()
    assert by_field["hidr"]["size_bytes"] == len(hidr_bytes)

    field_order = [entry["field"] for entry in entries]
    assert field_order == sorted(field_order)


def test_hash_input_files_skips_none_optionals_and_directory(tmp_path: Path) -> None:
    files = make_nw_files(tmp_path)  # optionals default to None

    entries = hash_input_files(files)
    present_fields = {entry["field"] for entry in entries}

    # The container ``directory`` field is never hashed.
    assert "directory" not in present_fields
    # Optional fields default to ``None`` and produce no entry.
    assert "modif" not in present_fields
    assert "cvar" not in present_fields
    assert "tratamento_fpha" not in present_fields
    # Required fields (always set) are present.
    assert "dger" in present_fields
    assert "vazoes" in present_fields


def test_hash_input_files_unreadable_file_records_none(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.dat"
    files = make_nw_files(tmp_path, dger=missing)

    entries = hash_input_files(files)
    dger_entry = next(entry for entry in entries if entry["field"] == "dger")

    assert dger_entry["sha256"] is None
    assert dger_entry["size_bytes"] is None
    assert dger_entry["path"] == str(missing)


def test_hash_input_files_hashes_decomp_files_and_skips_revision(
    tmp_path: Path,
) -> None:
    """``hash_input_files`` also serves ``DecompFiles``: the ``revision`` str
    field and the absent optionals are skipped; the three required paths are
    hashed."""
    from novomodelo_bridge.decomp.files import DecompFiles

    dadger_bytes = b"dadger-stub-\x00\x01"
    vazoes_bytes = b"vazoes-stub-\xff\xfe"
    hidr_bytes = b"hidr-stub-binary"
    dadger_path = tmp_path / "dadger.rv0"
    vazoes_path = tmp_path / "vazoes.rv0"
    hidr_path = tmp_path / "hidr.dat"
    dadger_path.write_bytes(dadger_bytes)
    vazoes_path.write_bytes(vazoes_bytes)
    hidr_path.write_bytes(hidr_bytes)

    files = DecompFiles(
        revision="rv0",
        dadger=dadger_path,
        vazoes=vazoes_path,
        hidr=hidr_path,
        dadgnl=None,
        renovaveis=None,
        polinjus=None,
    )

    entries = hash_input_files(files)
    by_field = {entry["field"]: entry for entry in entries}

    assert by_field["dadger"]["sha256"] == hashlib.sha256(dadger_bytes).hexdigest()
    assert by_field["dadger"]["size_bytes"] == len(dadger_bytes)
    assert by_field["vazoes"]["sha256"] == hashlib.sha256(vazoes_bytes).hexdigest()
    assert by_field["vazoes"]["size_bytes"] == len(vazoes_bytes)
    assert by_field["hidr"]["sha256"] == hashlib.sha256(hidr_bytes).hexdigest()
    assert by_field["hidr"]["size_bytes"] == len(hidr_bytes)

    present_fields = set(by_field)
    assert "revision" not in present_fields
    assert "dadgnl" not in present_fields
    assert "renovaveis" not in present_fields


def test_create_sets_version_git_and_timestamp() -> None:
    manifest = ConversionManifest.create(
        "convert newave",
        Path("src"),
        Path("dst"),
        entity_counts={"hydros": 2},
        input_files=[],
        diagnostics_summary={"warning": 1},
        diagnostics=[],
    )

    assert manifest.command == "convert newave"
    assert manifest.source_dir == "src"
    assert manifest.output_dir == "dst"
    assert manifest.bridge_version == novomodelo_bridge.__version__
    assert manifest.entity_counts == {"hydros": 2}
    assert manifest.diagnostics_summary == {"warning": 1}

    parsed = datetime.fromisoformat(manifest.timestamp)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None


def test_summarize_diagnostics_counts_by_severity_omitting_absent() -> None:
    diagnostics = [
        _diagnostic("a", Severity.INFO),
        _diagnostic("b", Severity.WARNING),
        _diagnostic("c", Severity.WARNING),
    ]

    summary = summarize_diagnostics(diagnostics)

    assert summary == {"info": 1, "warning": 2}
    assert "error" not in summary  # absent severities are omitted


def test_summarize_diagnostics_empty_is_empty_dict() -> None:
    assert summarize_diagnostics([]) == {}


def test_manifest_json_roundtrip(tmp_path: Path) -> None:
    manifest = ConversionManifest.create(
        "convert newave",
        Path("src"),
        Path("dst"),
        entity_counts={"hydros": 2, "thermals": 3},
        input_files=[
            {"field": "dger", "path": "dger.dat", "sha256": "deadbeef", "size_bytes": 4}
        ],
        diagnostics_summary={"warning": 1},
        diagnostics=[_diagnostic("x", Severity.WARNING).to_dict()],
    )

    path = tmp_path / "nested" / "conversion_manifest.json"
    manifest.to_json(path)

    assert path.exists()
    restored = ConversionManifest.from_json(path)
    assert restored == manifest


def test_to_json_creates_parent_dirs(tmp_path: Path) -> None:
    manifest = ConversionManifest.create(
        "convert newave",
        Path("src"),
        Path("dst"),
        entity_counts={},
        input_files=[],
        diagnostics_summary={},
        diagnostics=[],
    )
    path = tmp_path / "a" / "b" / "c" / "conversion_manifest.json"

    manifest.to_json(path)

    assert path.is_file()


def test_from_json_missing_raises_filenotfound(tmp_path: Path) -> None:
    missing = tmp_path / "absent" / "conversion_manifest.json"

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        ConversionManifest.from_json(missing)


def _base_manifest_data() -> dict[str, object]:
    """Return a minimal valid manifest JSON payload as a dict."""
    return {
        "command": "convert newave",
        "source_dir": "src",
        "output_dir": "dst",
        "bridge_version": "0.1.0",
        "git_sha": "abc1234",
        "timestamp": "2026-06-18T00:00:00+00:00",
    }


def test_from_json_ignores_unknown_field(tmp_path: Path) -> None:
    """A manifest with an extra unknown field loads, dropping the extra key.

    Forward compatibility with a newer bridge that writes additional fields.
    """
    data = _base_manifest_data()
    data["future_field"] = {"new": "thing"}
    path = tmp_path / "conversion_manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    manifest = ConversionManifest.from_json(path)

    assert manifest.command == "convert newave"
    assert not hasattr(manifest, "future_field")


def test_from_json_missing_optional_fields_uses_defaults(tmp_path: Path) -> None:
    """A manifest missing optional collection fields loads with empty defaults.

    Backward compatibility with an older manifest that omits ``entity_counts`` /
    ``input_files`` / ``diagnostics_summary`` / ``diagnostics``.
    """
    data = _base_manifest_data()
    path = tmp_path / "conversion_manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    manifest = ConversionManifest.from_json(path)

    assert manifest.entity_counts == {}
    assert manifest.input_files == []
    assert manifest.diagnostics_summary == {}
    assert manifest.diagnostics == []


@pytest.mark.skipif(not _in_git_checkout(), reason="not a git checkout")
def test_create_git_sha_in_repo() -> None:
    manifest = ConversionManifest.create(
        "convert newave",
        Path("src"),
        Path("dst"),
        entity_counts={},
        input_files=[],
        diagnostics_summary={},
        diagnostics=[],
    )

    assert manifest.git_sha is not None
    assert re.fullmatch(r"[0-9a-f]{7,}", manifest.git_sha) is not None


def test_git_sha_is_shared_not_duplicated() -> None:
    """Both manifests use the one shared git helper, never a duplicate."""
    from novomodelo_bridge.comparators.manifest import git_sha as comparison_git_sha
    from novomodelo_bridge.core.git import git_sha as shared_git_sha

    assert conversion_git_sha is shared_git_sha
    assert comparison_git_sha is shared_git_sha


def test_min_novomodelo_version_round_trip(tmp_path: Path) -> None:
    """``min_novomodelo_version`` survives a ``to_json`` / ``from_json`` round trip."""
    manifest = ConversionManifest.create(
        "convert newave",
        tmp_path,
        tmp_path,
        entity_counts={},
        input_files=[],
        diagnostics_summary={},
        diagnostics=[],
        min_novomodelo_version="0.9.1",
    )

    path = tmp_path / "conversion_manifest.json"
    manifest.to_json(path)
    restored = ConversionManifest.from_json(path)

    assert restored.min_novomodelo_version == "0.9.1"


def test_min_novomodelo_version_defaults_none() -> None:
    """``create`` without the kwarg leaves ``min_novomodelo_version`` ``None`` (EX-only)."""
    manifest = ConversionManifest.create(
        "convert newave",
        Path("src"),
        Path("dst"),
        entity_counts={},
        input_files=[],
        diagnostics_summary={},
        diagnostics=[],
    )

    assert manifest.min_novomodelo_version is None


def test_from_json_missing_min_novomodelo_version(tmp_path: Path) -> None:
    """An older manifest lacking ``min_novomodelo_version`` reads back as ``None``.

    Backward compatibility: a manifest written before the field existed loads
    without raising and defaults the field to ``None``.
    """
    data = _base_manifest_data()
    assert "min_novomodelo_version" not in data
    path = tmp_path / "conversion_manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    manifest = ConversionManifest.from_json(path)

    assert manifest.min_novomodelo_version is None
