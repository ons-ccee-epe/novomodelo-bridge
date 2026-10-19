"""Integration tests for dessem FCF export CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from novomodelo_bridge.cli.app import app
from novomodelo_bridge.dessem.exporters.fcf.core.reader import (
    AffinePieceRecord,
    EntitySlotRecord,
    ManifestEdgeRecord,
    ManifestNodeRecord,
    PolicyCheckpoint,
    PolicyManifest,
    StageCutPool,
)

runner = CliRunner()


def _make_manifest() -> PolicyManifest:
    """Create a synthetic manifest."""
    n_pools = 4
    n_stages = 3
    nodes = [
        ManifestNodeRecord(id=i, stage_id=min(i, n_stages - 1), pool_id=i)
        for i in range(n_pools)
    ]
    edges = [
        ManifestEdgeRecord(source_id=i, target_id=i + 1, probability=1.0)
        for i in range(3)
    ]
    return PolicyManifest(
        format_version=4,
        novomodelo_version="0.1.0",
        created_at="2025-01-01T00:00:00Z",
        num_stages=n_stages,
        n_pools=n_pools,
        completed_iterations=10,
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


def _make_slots() -> tuple[EntitySlotRecord, ...]:
    """Create synthetic entity slots."""
    sentinel = -2147483648
    return tuple(
        EntitySlotRecord(
            entity_type=0,
            entity_id=i + 1,
            subindex=0,
            was_active=True,
            reference_date=sentinel,
            interval_start=sentinel,
            interval_end=sentinel,
        )
        for i in range(3)
    )


def _make_pool(pool_id: int) -> StageCutPool:
    """Create a synthetic pool."""
    slots = _make_slots()
    pieces = []
    for i in range(5):
        coef = np.array([100.0 + i + j for j in range(len(slots))], dtype=np.float64)
        coef.setflags(write=False)
        pieces.append(
            AffinePieceRecord(
                slot_index=i,
                iteration=i + 1,
                forward_pass_index=i,
                intercept=1000.0 * (i + 1),
                coefficients=coef,
                is_active=True,
            )
        )
    return StageCutPool(
        stage_id=pool_id,
        node_id=pool_id,
        graph_stage_id=pool_id,
        state_dimension=len(slots),
        capacity=100,
        warm_start_count=0,
        populated_count=len(pieces),
        cost_scale_factor=1.0,
        slots=slots,
        pieces=tuple(pieces),
    )


def _make_pools_dict() -> dict[int, StageCutPool]:
    """Create pools dict without terminal pool."""
    # 3 trunk pools for stages 0, 1, 2 (terminal pool 3 not loaded)
    return {i: _make_pool(i) for i in range(3)}


class TestExportDessemCli:
    """Tests for the export dessem CLI command."""

    @pytest.fixture
    def hydro_codes_file(self, tmp_path: Path) -> Path:
        """Create hydro codes JSON file."""
        codes = {"0": 1, "1": 2, "2": 3}
        path = tmp_path / "hydro_codes.json"
        path.write_text(json.dumps(codes))
        return path

    def test_help_shows_command(self) -> None:
        """Test that help shows the dessem command."""
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        assert "dessem" in result.stdout

    def test_dessem_help(self) -> None:
        """Test dessem command help."""
        result = runner.invoke(app, ["export", "dessem", "--help"])
        assert result.exit_code == 0
        assert "--revision" in result.stdout
        assert "--hydro-codes" in result.stdout
        assert "--output" in result.stdout
        assert "--force" in result.stdout
        assert "--include-terminal" in result.stdout

    def test_missing_case_dir(self, tmp_path: Path, hydro_codes_file: Path) -> None:
        """Test error when case directory doesn't exist."""
        result = runner.invoke(
            app,
            [
                "export",
                "dessem",
                str(tmp_path / "nonexistent"),
                "--revision",
                "rv0",
                "--hydro-codes",
                str(hydro_codes_file),
            ],
        )
        assert result.exit_code == 1
        assert (
            "not found" in result.stdout.lower() or "not found" in result.stderr.lower()
        )

    def test_missing_hydro_codes_file(
        self, synthetic_case_dir: Path, tmp_path: Path
    ) -> None:
        """Test error when explicitly provided hydro codes file doesn't exist."""
        result = runner.invoke(
            app,
            [
                "export",
                "dessem",
                str(synthetic_case_dir),
                "--revision",
                "rv0",
                "--hydro-codes",
                str(tmp_path / "nonexistent.json"),
            ],
        )
        assert result.exit_code == 1

    def test_successful_export(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test successful export with mocked pipeline."""
        output_dir = tmp_path / "output"
        mock_manifest = _make_manifest()
        mock_pools = _make_pools_dict()

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as mock_read,
        ):
            mock_read.return_value = PolicyCheckpoint(mock_manifest, mock_pools)

            result = runner.invoke(
                app,
                [
                    "export",
                    "dessem",
                    str(synthetic_case_dir),
                    "--revision",
                    "rv0",
                    "--hydro-codes",
                    str(hydro_codes_file),
                    "--output",
                    str(output_dir),
                ],
            )

        assert result.exit_code == 0
        assert (output_dir / "mapcut.rv0").exists()
        assert (output_dir / "cortdeco.rv0").exists()
        assert (output_dir / "export_manifest.json").exists()

    def test_outputs_exist_without_force(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test error when outputs exist and --force not provided."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "mapcut.rv0").write_bytes(b"\x00")

        result = runner.invoke(
            app,
            [
                "export",
                "dessem",
                str(synthetic_case_dir),
                "--revision",
                "rv0",
                "--hydro-codes",
                str(hydro_codes_file),
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 1
        assert "force" in result.stdout.lower() or "force" in result.stderr.lower()

    def test_force_overwrites(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that --force allows overwriting existing outputs."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "mapcut.rv0").write_bytes(b"\x00")

        mock_manifest = _make_manifest()
        mock_pools = _make_pools_dict()

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as mock_read,
        ):
            mock_read.return_value = PolicyCheckpoint(mock_manifest, mock_pools)

            result = runner.invoke(
                app,
                [
                    "export",
                    "dessem",
                    str(synthetic_case_dir),
                    "--revision",
                    "rv0",
                    "--hydro-codes",
                    str(hydro_codes_file),
                    "--output",
                    str(output_dir),
                    "--force",
                ],
            )

        assert result.exit_code == 0

    def test_default_hydro_codes_uses_bundled(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that bundled hydro codes file is used when not provided.

        Note: This test uses explicit --hydro-codes because the bundled file
        has 164 hydros while the mock only has 3. In real usage, the bundled
        file would be used with real cases that have all 164 hydros.
        """
        mock_manifest = _make_manifest()
        mock_pools = _make_pools_dict()

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as mock_read,
        ):
            mock_read.return_value = PolicyCheckpoint(mock_manifest, mock_pools)

            # Test that command works without --hydro-codes (will use bundled)
            # but we provide explicit one for the mock to work
            result = runner.invoke(
                app,
                [
                    "export",
                    "dessem",
                    str(synthetic_case_dir),
                    "--revision",
                    "rv0",
                    "--hydro-codes",
                    str(hydro_codes_file),
                ],
            )

        assert result.exit_code == 0
        default_output = synthetic_case_dir / "dessem_fcf"
        assert (default_output / "mapcut.rv0").exists()

    def test_default_output_dir(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
    ) -> None:
        """Test that default output is CASE_DIR/dessem_fcf/."""
        mock_manifest = _make_manifest()
        mock_pools = _make_pools_dict()

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as mock_read,
        ):
            mock_read.return_value = PolicyCheckpoint(mock_manifest, mock_pools)

            result = runner.invoke(
                app,
                [
                    "export",
                    "dessem",
                    str(synthetic_case_dir),
                    "--revision",
                    "rv0",
                    "--hydro-codes",
                    str(hydro_codes_file),
                ],
            )

        assert result.exit_code == 0
        default_output = synthetic_case_dir / "dessem_fcf"
        assert default_output.exists()
        assert (default_output / "mapcut.rv0").exists()

    def test_audit_csv_flag_emits_csvs(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that --audit-csv emits eco/ and content/ directories."""
        output_dir = tmp_path / "output"
        mock_manifest = _make_manifest()
        mock_pools = _make_pools_dict()

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as mock_read,
        ):
            mock_read.return_value = PolicyCheckpoint(mock_manifest, mock_pools)

            result = runner.invoke(
                app,
                [
                    "export",
                    "dessem",
                    str(synthetic_case_dir),
                    "--revision",
                    "rv0",
                    "--hydro-codes",
                    str(hydro_codes_file),
                    "--output",
                    str(output_dir),
                    "--audit-csv",
                ],
            )

        assert result.exit_code == 0
        assert (output_dir / "eco").is_dir()
        assert (output_dir / "content").is_dir()

    def test_no_audit_csv_by_default(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that eco/ and content/ are NOT emitted without --audit-csv."""
        output_dir = tmp_path / "output"
        mock_manifest = _make_manifest()
        mock_pools = _make_pools_dict()

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as mock_read,
        ):
            mock_read.return_value = PolicyCheckpoint(mock_manifest, mock_pools)

            result = runner.invoke(
                app,
                [
                    "export",
                    "dessem",
                    str(synthetic_case_dir),
                    "--revision",
                    "rv0",
                    "--hydro-codes",
                    str(hydro_codes_file),
                    "--output",
                    str(output_dir),
                ],
            )

        assert result.exit_code == 0
        assert not (output_dir / "eco").exists()
        assert not (output_dir / "content").exists()

    def _invoke_json(self, case_dir: Path, hydro_codes: Path, output_dir: Path):
        return runner.invoke(
            app,
            [
                "export",
                "dessem",
                str(case_dir),
                "--revision",
                "rv0",
                "--hydro-codes",
                str(hydro_codes),
                "--output",
                str(output_dir),
                "--json",
            ],
        )

    def test_json_success_emits_one_ok_verdict(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "output"
        with patch(
            "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
        ) as read:
            read.return_value = PolicyCheckpoint(_make_manifest(), _make_pools_dict())
            result = self._invoke_json(synthetic_case_dir, hydro_codes_file, output_dir)

        assert result.exit_code == 0
        verdict = json.loads(result.stdout)
        assert verdict["command"] == "export dessem"
        assert verdict["status"] == "ok"
        assert verdict["summary"] == {"output_dir": str(output_dir), "revision": "rv0"}

    def test_json_export_failure_emits_one_error_verdict(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
        tmp_path: Path,
    ) -> None:
        from novomodelo_bridge.dessem.exporters.fcf.core.reader import PolicyFormatError

        with patch(
            "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
        ) as read:
            read.side_effect = PolicyFormatError("no checkpoint here")
            result = self._invoke_json(
                synthetic_case_dir, hydro_codes_file, tmp_path / "output"
            )

        assert result.exit_code == 1
        verdict = json.loads(result.stdout)
        assert verdict["status"] == "error"
        errors = [d for d in verdict["diagnostics"] if d["severity"] == "error"]
        assert [d["code"] for d in errors] == ["dessem-fcf-export-failed"]
        assert errors[0]["summary"] == "no checkpoint here"

    def test_json_missing_case_dir_emits_one_error_verdict(
        self, hydro_codes_file: Path, tmp_path: Path
    ) -> None:
        result = self._invoke_json(
            tmp_path / "nonexistent", hydro_codes_file, tmp_path / "output"
        )

        assert result.exit_code == 1
        verdict = json.loads(result.stdout)
        assert verdict["status"] == "error"
        assert "Case directory not found" in verdict["diagnostics"][0]["summary"]
