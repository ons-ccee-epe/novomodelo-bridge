"""Integration tests for dessem/exporters/fcf/pipeline.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from novomodelo_bridge.dessem.exporters.fcf.cross_check import (
    reject_pair,
)
from novomodelo_bridge.dessem.exporters.fcf.pipeline import ExportSettings, run_export
from novomodelo_bridge.dessem.exporters.fcf.reader import (
    AffinePieceRecord,
    EntitySlotRecord,
    ManifestEdgeRecord,
    ManifestNodeRecord,
    PolicyManifest,
    StageCutPool,
)


def _make_pool(
    pool_id: int,
    node_id: int,
    graph_stage_id: int,
    slots: tuple[EntitySlotRecord, ...],
    n_pieces: int,
    warm_start: int = 0,
) -> StageCutPool:
    """Create a synthetic cut pool."""
    pieces = []
    n_slots = len(slots)
    for i in range(n_pieces):
        coef = np.array([100.0 + i + j for j in range(n_slots)], dtype=np.float64)
        coef.setflags(write=False)
        pieces.append(
            AffinePieceRecord(
                piece_id=i,
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
        node_id=node_id,
        graph_stage_id=graph_stage_id,
        state_dimension=n_slots,
        capacity=100,
        warm_start_count=warm_start,
        populated_count=n_pieces,
        cost_scale_factor=1.0,
        slots=slots,
        pieces=tuple(pieces),
        active_cut_indices=tuple(range(n_pieces)),
    )


def _make_manifest(n_stages: int = 3, n_pools: int = 4) -> PolicyManifest:
    """Create a synthetic manifest with simple tree structure.

    Creates n_pools pools with n_stages stages. The last pool is the terminal.
    """
    nodes = [
        ManifestNodeRecord(id=i, stage_id=min(i, n_stages - 1), pool_id=i)
        for i in range(n_pools)
    ]
    edges = [
        ManifestEdgeRecord(source_id=i, target_id=i + 1, probability=1.0)
        for i in range(n_pools - 1)
    ]
    return PolicyManifest(
        format_version=1,
        novomodelo_version="0.1.0",
        created_at="2025-01-01T00:00:00Z",
        num_stages=n_stages,
        n_pools=n_pools,
        completed_iterations=10,
        cost_scale_factor=1.0,
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


def _make_slots(n_hydros: int) -> tuple[EntitySlotRecord, ...]:
    """Create synthetic entity slots for hydros only (no GNL)."""
    sentinel = -2147483648
    return tuple(
        EntitySlotRecord(
            entity_type=0,
            entity_id=i + 1,
            subindex=0,
            was_active=True,
            delivery_date=sentinel,
        )
        for i in range(n_hydros)
    )


def _make_pools_dict(n_stages: int = 3) -> dict[int, StageCutPool]:
    """Create pools dict without terminal pool (simulating load_pools behavior)."""
    slots = _make_slots(3)
    # 3 trunk pools for stages 0, 1, 2 (terminal pool 3 not included)
    return {i: _make_pool(i, i, i, slots, n_pieces=5) for i in range(n_stages)}


class TestPipelineIntegration:
    """Integration tests for the full export pipeline."""

    @pytest.fixture
    def hydro_codes_file(self, tmp_path: Path) -> Path:
        """Create hydro codes JSON file."""
        codes = {"0": 1, "1": 2, "2": 3}
        path = tmp_path / "hydro_codes.json"
        path.write_text(json.dumps(codes))
        return path

    @pytest.fixture
    def mock_manifest(self) -> PolicyManifest:
        """Create a mock policy manifest."""
        return _make_manifest(n_stages=3, n_pools=4)

    @pytest.fixture
    def mock_pools(self) -> dict[int, StageCutPool]:
        """Create mock pools for testing (without terminal)."""
        return _make_pools_dict(n_stages=3)

    def test_export_produces_files(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
        mock_manifest: PolicyManifest,
        mock_pools: dict[int, StageCutPool],
        tmp_path: Path,
    ) -> None:
        """Test that export produces expected files."""
        output_dir = tmp_path / "dessem_fcf"

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy_manifest"
            ) as mock_read_manifest,
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.load_pools"
            ) as mock_load,
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.nodes_by_pool"
            ) as mock_nodes,
        ):
            mock_read_manifest.return_value = mock_manifest
            mock_load.return_value = mock_pools
            # Pool IDs 0,1,2,3 (terminal is 3)
            mock_nodes.return_value = {i: (i,) for i in range(4)}

            settings = ExportSettings(
                eco_subdirectory="eco",
                content_subdirectory="content",
                hydro_codes_path=str(hydro_codes_file),
                include_terminal_pool=False,
            )

            result = run_export(
                case_dir=synthetic_case_dir,
                revision="rv0",
                output_dir=output_dir,
                settings=settings,
            )

        assert result == 0
        assert (output_dir / "mapcut.rv0").exists()
        assert (output_dir / "cortdeco.rv0").exists()
        assert (output_dir / "export_manifest.json").exists()

    def test_manifest_contains_metadata(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
        mock_manifest: PolicyManifest,
        mock_pools: dict[int, StageCutPool],
        tmp_path: Path,
    ) -> None:
        """Test that export manifest contains required metadata."""
        output_dir = tmp_path / "dessem_fcf"

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy_manifest"
            ) as mock_read_manifest,
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.load_pools"
            ) as mock_load,
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.nodes_by_pool"
            ) as mock_nodes,
        ):
            mock_read_manifest.return_value = mock_manifest
            mock_load.return_value = mock_pools
            mock_nodes.return_value = {i: (i,) for i in range(4)}

            settings = ExportSettings(
                eco_subdirectory="eco",
                content_subdirectory="content",
                hydro_codes_path=str(hydro_codes_file),
                include_terminal_pool=False,
            )

            run_export(
                case_dir=synthetic_case_dir,
                revision="rv0",
                output_dir=output_dir,
                settings=settings,
            )

        manifest_path = output_dir / "export_manifest.json"
        manifest: dict[str, Any] = json.loads(manifest_path.read_text())

        assert manifest["status"] == "ok"
        assert manifest["revision"] == "rv0"
        assert "premises" in manifest
        assert len(manifest["premises"]) == 15
        assert "outputs" in manifest
        assert "mapcut" in manifest["outputs"]
        assert "cortdeco" in manifest["outputs"]

    def test_eco_csvs_created(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
        mock_manifest: PolicyManifest,
        mock_pools: dict[int, StageCutPool],
        tmp_path: Path,
    ) -> None:
        """Test that ECO CSVs are created."""
        output_dir = tmp_path / "dessem_fcf"

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy_manifest"
            ) as mock_read_manifest,
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.load_pools"
            ) as mock_load,
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.nodes_by_pool"
            ) as mock_nodes,
        ):
            mock_read_manifest.return_value = mock_manifest
            mock_load.return_value = mock_pools
            mock_nodes.return_value = {i: (i,) for i in range(4)}

            settings = ExportSettings(
                eco_subdirectory="eco",
                content_subdirectory="content",
                hydro_codes_path=str(hydro_codes_file),
                include_terminal_pool=False,
            )

            run_export(
                case_dir=synthetic_case_dir,
                revision="rv0",
                output_dir=output_dir,
                settings=settings,
            )

        eco_dir = output_dir / "eco"
        assert eco_dir.exists()
        # At least one ECO CSV should exist
        csv_files = list(eco_dir.glob("*.csv"))
        assert len(csv_files) > 0

    def test_content_csvs_created(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
        mock_manifest: PolicyManifest,
        mock_pools: dict[int, StageCutPool],
        tmp_path: Path,
    ) -> None:
        """Test that content CSVs are created."""
        output_dir = tmp_path / "dessem_fcf"

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy_manifest"
            ) as mock_read_manifest,
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.load_pools"
            ) as mock_load,
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.nodes_by_pool"
            ) as mock_nodes,
        ):
            mock_read_manifest.return_value = mock_manifest
            mock_load.return_value = mock_pools
            mock_nodes.return_value = {i: (i,) for i in range(4)}

            settings = ExportSettings(
                eco_subdirectory="eco",
                content_subdirectory="content",
                hydro_codes_path=str(hydro_codes_file),
                include_terminal_pool=False,
            )

            run_export(
                case_dir=synthetic_case_dir,
                revision="rv0",
                output_dir=output_dir,
                settings=settings,
            )

        content_dir = output_dir / "content"
        assert content_dir.exists()
        assert (content_dir / "mapcut_content.csv").exists()
        assert (content_dir / "cortdeco_content.csv").exists()


class TestCrossCheckIntegration:
    """Integration tests for cross-check and pair rejection."""

    @pytest.fixture
    def matched_pair(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create a matched mapcut/cortdeco pair."""
        mapcut = tmp_path / "mapcut.rv0"
        cortdeco = tmp_path / "cortdeco.rv0"

        # Create minimal valid files
        # mapcut: at least RECORD_SIZE bytes
        mapcut.write_bytes(b"\x00" * 48020)
        # cortdeco: at least TAMANHO_CORTE bytes
        cortdeco.write_bytes(b"\x00" * 26976)

        return mapcut, cortdeco

    def test_reject_pair_renames_files(self, matched_pair: tuple[Path, Path]) -> None:
        """Test that reject_pair renames both files."""
        mapcut, cortdeco = matched_pair

        reject_pair(mapcut, cortdeco)

        assert not mapcut.exists()
        assert not cortdeco.exists()
        assert mapcut.with_suffix(".rv0.rejected").exists()
        assert cortdeco.with_suffix(".rv0.rejected").exists()

    def test_reject_pair_handles_missing_mapcut(self, tmp_path: Path) -> None:
        """Test reject_pair when mapcut doesn't exist."""
        mapcut = tmp_path / "mapcut.rv0"
        cortdeco = tmp_path / "cortdeco.rv0"
        cortdeco.write_bytes(b"\x00" * 100)

        # Should not raise
        reject_pair(mapcut, cortdeco)

        assert cortdeco.with_suffix(".rv0.rejected").exists()

    def test_reject_pair_handles_missing_cortdeco(self, tmp_path: Path) -> None:
        """Test reject_pair when cortdeco doesn't exist."""
        mapcut = tmp_path / "mapcut.rv0"
        cortdeco = tmp_path / "cortdeco.rv0"
        mapcut.write_bytes(b"\x00" * 100)

        # Should not raise
        reject_pair(mapcut, cortdeco)

        assert mapcut.with_suffix(".rv0.rejected").exists()


class TestFailureManifest:
    """Integration tests for failure manifest creation."""

    @pytest.fixture
    def hydro_codes_file(self, tmp_path: Path) -> Path:
        """Create hydro codes JSON file."""
        codes = {"0": 1, "1": 2, "2": 3}
        path = tmp_path / "hydro_codes.json"
        path.write_text(json.dumps(codes))
        return path

    def test_failure_manifest_on_format_error(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that failure manifest is written on PolicyFormatError."""
        from novomodelo_bridge.dessem.exporters.fcf.reader import PolicyFormatError

        output_dir = tmp_path / "dessem_fcf"

        with patch(
            "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy_manifest"
        ) as mock_read:
            mock_read.side_effect = PolicyFormatError("test error")

            settings = ExportSettings(
                eco_subdirectory="eco",
                content_subdirectory="content",
                hydro_codes_path=str(hydro_codes_file),
                include_terminal_pool=False,
            )

            result = run_export(
                case_dir=synthetic_case_dir,
                revision="rv0",
                output_dir=output_dir,
                settings=settings,
            )

        assert result == 1
        manifest_path = output_dir / "export_manifest.json"
        assert manifest_path.exists()

        manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        assert manifest["status"] == "failed"
        assert "failed_step" in manifest

    def test_failure_manifest_on_mapping_error(
        self,
        synthetic_case_dir: Path,
        hydro_codes_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that failure manifest is written on MappingError."""
        from novomodelo_bridge.dessem.exporters.fcf.mapper import MappingError

        output_dir = tmp_path / "dessem_fcf"
        mock_manifest = _make_manifest()
        mock_pools = {
            i: _make_pool(i, i, i, _make_slots(3), n_pieces=5) for i in range(4)
        }

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy_manifest"
            ) as mock_read_manifest,
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.load_pools"
            ) as mock_load,
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.nodes_by_pool"
            ) as mock_nodes,
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.assert_no_travel_time"
            ) as mock_travel,
        ):
            mock_read_manifest.return_value = mock_manifest
            mock_load.return_value = mock_pools
            mock_nodes.return_value = {i: (i,) for i in range(4)}
            mock_travel.side_effect = MappingError("travel time not supported")

            settings = ExportSettings(
                eco_subdirectory="eco",
                content_subdirectory="content",
                hydro_codes_path=str(hydro_codes_file),
                include_terminal_pool=False,
            )

            result = run_export(
                case_dir=synthetic_case_dir,
                revision="rv0",
                output_dir=output_dir,
                settings=settings,
            )

        assert result == 1
        manifest_path = output_dir / "export_manifest.json"
        assert manifest_path.exists()

        manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        assert manifest["status"] == "failed"
