"""Tests for :mod:`novomodelo_bridge.comparators.manifest`."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

import novomodelo_bridge
from novomodelo_bridge.comparators.manifest import ComparisonManifest
from novomodelo_bridge.core.git import git_sha


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


def test_create_sets_bridge_version_and_timestamp() -> None:
    manifest = ComparisonManifest.create("compare newave", Path("nw"), Path("cb"), 1e-2)

    assert manifest.command == "compare newave"
    assert manifest.source_dir == "nw"
    assert manifest.novomodelo_output_dir == "cb"
    assert manifest.tolerance == 1e-2
    assert manifest.bridge_version == novomodelo_bridge.__version__
    assert manifest.novomodelo_version is None
    assert manifest.newave_version is None
    assert manifest.artifacts == []
    assert manifest.top_divergences == []
    assert manifest.input_files == []
    assert manifest.diagnostics_summary == {}
    assert manifest.diagnostics == []

    parsed = datetime.fromisoformat(manifest.timestamp)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None


def test_create_threads_through_input_files_and_diagnostics() -> None:
    """``create``'s new keyword-only fields default empty and thread through."""
    input_files = [
        {"field": "dadger", "path": "dadger.rv0", "sha256": "abc", "size_bytes": 4}
    ]
    diagnostics = [{"code": "x", "severity": "warning"}]

    manifest = ComparisonManifest.create(
        "compare decomp",
        Path("decomp"),
        Path("cb"),
        1e-2,
        input_files=input_files,
        diagnostics_summary={"warning": 1},
        diagnostics=diagnostics,
    )

    assert manifest.input_files == input_files
    assert manifest.diagnostics_summary == {"warning": 1}
    assert manifest.diagnostics == diagnostics


def test_create_passes_through_optional_versions() -> None:
    manifest = ComparisonManifest.create(
        "compare bounds",
        Path("nw"),
        Path("cb"),
        0.05,
        novomodelo_version="1.2.3",
        newave_version="28.0",
    )

    assert manifest.command == "compare bounds"
    assert manifest.novomodelo_version == "1.2.3"
    assert manifest.newave_version == "28.0"


@pytest.mark.skipif(not _in_git_checkout(), reason="not a git checkout")
def test_create_git_sha_in_repo() -> None:
    manifest = ComparisonManifest.create("compare newave", Path("nw"), Path("cb"), 1e-2)

    assert manifest.git_sha is not None
    assert re.fullmatch(r"[0-9a-f]{7,}", manifest.git_sha) is not None


def test_git_sha_returns_none_when_git_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)

    assert git_sha() is None


def test_git_sha_returns_none_on_nonzero_returncode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["git"], returncode=128, stdout="", stderr="not a git repo"
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    assert git_sha() is None


@pytest.mark.skipif(not _in_git_checkout(), reason="not a git checkout")
def test_git_sha_uses_bridge_repo_not_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``git_sha`` must resolve the bridge repo, not the caller's cwd."""
    bridge_dir = Path(novomodelo_bridge.__file__).resolve().parent
    expected = subprocess.run(
        ["git", "-C", str(bridge_dir), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    monkeypatch.chdir(tmp_path)
    result = git_sha()

    assert result is not None
    assert result == expected


def test_manifest_json_roundtrip(tmp_path: Path) -> None:
    manifest = ComparisonManifest.create(
        "compare newave",
        Path("nw"),
        Path("cb"),
        1e-2,
        novomodelo_version="1.2.3",
        newave_version="28.0",
    )
    manifest.artifacts = ["report.html", "dataset.parquet"]
    manifest.top_divergences = [
        {"entity": "hydro_42", "metric": "spot_price", "delta": 0.21}
    ]

    path = tmp_path / "nested" / "comparison.json"
    manifest.to_json(path)

    assert path.exists()
    restored = ComparisonManifest.from_json(path)
    assert restored == manifest


def test_from_json_missing_raises_filenotfound(tmp_path: Path) -> None:
    missing = tmp_path / "absent" / "comparison.json"

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        ComparisonManifest.from_json(missing)


def test_to_json_creates_parent_dirs(tmp_path: Path) -> None:
    manifest = ComparisonManifest.create("compare bounds", Path("nw"), Path("cb"), 1e-3)
    path = tmp_path / "a" / "b" / "c" / "comparison.json"

    manifest.to_json(path)

    assert path.is_file()


def _base_manifest_data() -> dict[str, object]:
    """Return a minimal valid manifest JSON payload as a dict."""
    return {
        "command": "compare newave",
        "source_dir": "nw",
        "novomodelo_output_dir": "cb",
        "tolerance": 1e-2,
        "bridge_version": "0.1.0",
        "git_sha": "abc1234",
        "timestamp": "2026-06-13T00:00:00+00:00",
    }


def test_from_json_ignores_unknown_field(tmp_path: Path) -> None:
    """A manifest with an extra unknown field loads, dropping the extra key.

    Regression for FINDING 3: forward compatibility with a newer bridge that
    writes additional fields.
    """
    data = _base_manifest_data()
    data["future_field"] = {"new": "thing"}
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    manifest = ComparisonManifest.from_json(path)

    assert manifest.command == "compare newave"
    assert not hasattr(manifest, "future_field")


def test_from_json_missing_optional_fields_uses_defaults(tmp_path: Path) -> None:
    """A manifest missing optional list fields loads with empty-list defaults.

    Regression for FINDING 3: backward compatibility with an older manifest that
    omits ``artifacts`` / ``top_divergences``.
    """
    data = _base_manifest_data()
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    manifest = ComparisonManifest.from_json(path)

    assert manifest.artifacts == []
    assert manifest.top_divergences == []
    assert manifest.novomodelo_version is None
    assert manifest.newave_version is None
    assert manifest.input_files == []
    assert manifest.diagnostics_summary == {}
    assert manifest.diagnostics == []
