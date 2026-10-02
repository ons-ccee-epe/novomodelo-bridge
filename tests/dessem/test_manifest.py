"""Tests for dessem/exporters/fcf/manifest.py."""

import json
from pathlib import Path

from novomodelo_bridge.dessem.exporters.fcf.manifest import (
    build_export_manifest,
    write_export_manifest,
)
from novomodelo_bridge.dessem.exporters.fcf.premises import PREMISES


class TestBuildExportManifest:
    """Test export manifest building."""

    def test_builds_valid_manifest(self, tmp_path: Path) -> None:
        manifest = build_export_manifest(
            case_path=tmp_path / "case",
            revision="rv0",
            settings_snapshot={"key": "value"},
            outputs={"mapcut": tmp_path / "mapcut.rv0"},
        )

        assert manifest.revision == "rv0"
        assert manifest.status == "ok"
        assert manifest.failed_step is None

    def test_includes_all_premises(self, tmp_path: Path) -> None:
        manifest = build_export_manifest(
            case_path=tmp_path / "case",
            revision="rv0",
            settings_snapshot={},
            outputs={},
        )

        assert manifest.premises == PREMISES
        assert len(manifest.premises) == 15

    def test_failure_manifest(self, tmp_path: Path) -> None:
        manifest = build_export_manifest(
            case_path=tmp_path / "case",
            revision="rv0",
            settings_snapshot={},
            outputs={},
            status="failed",
            failed_step="write cortdeco",
        )

        assert manifest.status == "failed"
        assert manifest.failed_step == "write cortdeco"

    def test_includes_library_versions(self, tmp_path: Path) -> None:
        manifest = build_export_manifest(
            case_path=tmp_path / "case",
            revision="rv0",
            settings_snapshot={},
            outputs={},
        )

        assert "numpy" in manifest.library_versions
        assert "flatbuffers" in manifest.library_versions

    def test_created_at_is_iso8601(self, tmp_path: Path) -> None:
        manifest = build_export_manifest(
            case_path=tmp_path / "case",
            revision="rv0",
            settings_snapshot={},
            outputs={},
        )

        # Should parse without error
        from datetime import datetime

        datetime.fromisoformat(manifest.created_at.replace("Z", "+00:00"))


class TestWriteExportManifest:
    """Test export manifest writing."""

    def test_writes_json_file(self, tmp_path: Path) -> None:
        manifest = build_export_manifest(
            case_path=tmp_path / "case",
            revision="rv0",
            settings_snapshot={},
            outputs={},
        )
        path = tmp_path / "export_manifest.json"

        write_export_manifest(manifest, path)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["revision"] == "rv0"

    def test_json_is_sorted(self, tmp_path: Path) -> None:
        manifest = build_export_manifest(
            case_path=tmp_path / "case",
            revision="rv0",
            settings_snapshot={"z_key": 1, "a_key": 2},
            outputs={},
        )
        path = tmp_path / "export_manifest.json"

        write_export_manifest(manifest, path)

        content = path.read_text()
        # Keys should be alphabetically sorted
        assert content.index('"a_key"') < content.index('"z_key"')

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        manifest = build_export_manifest(
            case_path=tmp_path / "case",
            revision="rv0",
            settings_snapshot={},
            outputs={},
        )
        path = tmp_path / "nested" / "dir" / "export_manifest.json"

        write_export_manifest(manifest, path)

        assert path.exists()

    def test_no_partial_file_on_success(self, tmp_path: Path) -> None:
        manifest = build_export_manifest(
            case_path=tmp_path / "case",
            revision="rv0",
            settings_snapshot={},
            outputs={},
        )
        path = tmp_path / "export_manifest.json"

        write_export_manifest(manifest, path)

        partial = path.with_name(path.name + ".partial")
        assert not partial.exists()
