"""Integration tests for dessem/exporters/fcf/pipeline.py."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.dessem.exporters.fcf.core.cross_check import (
    cross_check_pair,
    read_mapcut,
    reject_pair,
)
from novomodelo_bridge.dessem.exporters.fcf.core.layout import TAMANHO_CORTE
from novomodelo_bridge.dessem.exporters.fcf.core.mapper import SourceHydro
from novomodelo_bridge.dessem.exporters.fcf.core.reader import (
    UNDATED,
    AffinePieceRecord,
    EntitySlotRecord,
    ManifestEdgeRecord,
    ManifestNodeRecord,
    PolicyCheckpoint,
    PolicyManifest,
    StageCutPool,
)
from novomodelo_bridge.dessem.exporters.fcf.pipeline import ExportSettings, run_export
from novomodelo_bridge.dessem.exporters.fcf.source import SourceDeck


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
        format_version=4,
        novomodelo_version="0.1.0",
        created_at="2025-01-01T00:00:00Z",
        num_stages=n_stages,
        n_pools=n_pools,
        completed_iterations=10,
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


def _make_slots(
    n_hydros: int, entity_ids: tuple[int, ...] | None = None
) -> tuple[EntitySlotRecord, ...]:
    """Storage slots only (no GNL), for hydro ids 1..n unless *entity_ids* says."""
    sentinel = -2147483648
    ids = entity_ids if entity_ids is not None else tuple(range(1, n_hydros + 1))
    return tuple(
        EntitySlotRecord(
            entity_type=0,
            entity_id=entity_id,
            subindex=0,
            was_active=True,
            reference_date=sentinel,
            interval_start=sentinel,
            interval_end=sentinel,
        )
        for entity_id in ids
    )


def _make_pools_dict(n_stages: int = 3) -> dict[int, StageCutPool]:
    """Create the trunk pools; the terminal pool 3 is never selected."""
    slots = _make_slots(3)
    # 3 trunk pools for stages 0, 1, 2 (terminal pool 3 not included)
    return {i: _make_pool(i, i, i, slots, n_pieces=5) for i in range(n_stages)}


class TestPipelineIntegration:
    """Integration tests for the full export pipeline."""

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
        synthetic_source: SourceDeck,
        mock_manifest: PolicyManifest,
        mock_pools: dict[int, StageCutPool],
        tmp_path: Path,
    ) -> None:
        """Test that export produces expected files."""
        output_dir = tmp_path / "dessem_fcf"

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as mock_read,
        ):
            mock_read.return_value = PolicyCheckpoint(mock_manifest, mock_pools)

            settings = ExportSettings(
                eco_subdirectory="eco",
                content_subdirectory="content",
                include_terminal_pool=False,
            )

            result = run_export(
                case_dir=synthetic_case_dir,
                source=synthetic_source,
                output_dir=output_dir,
                settings=settings,
            )

        assert result == 0
        assert (output_dir / "mapcut.rv0").exists()
        assert (output_dir / "cortdeco.rv0").exists()
        assert (output_dir / "export_manifest.json").exists()

    def test_warm_start_cuts_are_not_written(
        self,
        synthetic_case_dir: Path,
        synthetic_source: SourceDeck,
        mock_manifest: PolicyManifest,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "dessem_fcf"
        slots = _make_slots(3)
        pools = {
            i: _make_pool(i, i, i, slots, n_pieces=5, warm_start=1) for i in range(3)
        }

        with patch(
            "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
        ) as read:
            read.return_value = PolicyCheckpoint(mock_manifest, pools)
            result = run_export(
                case_dir=synthetic_case_dir,
                source=synthetic_source,
                output_dir=output_dir,
                settings=ExportSettings(),
            )

        assert result == 0
        # 3 pools x (5 populated - 1 warm-start) cuts, plus the trailing record.
        cuts_written = 3 * 4 + 1
        assert (output_dir / "cortdeco.rv0").stat().st_size == (
            cuts_written * TAMANHO_CORTE
        )

    def test_records_carry_deck_codes_and_downstream_positions(
        self,
        synthetic_case_dir: Path,
        synthetic_source: SourceDeck,
        mock_manifest: PolicyManifest,
        mock_pools: dict[int, StageCutPool],
        tmp_path: Path,
    ) -> None:
        # The synthetic cascade sends UHE_B and UHE_C into UHE_A, the first plant.
        output_dir = tmp_path / "dessem_fcf"

        with patch(
            "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
        ) as read:
            read.return_value = PolicyCheckpoint(mock_manifest, mock_pools)
            result = run_export(
                case_dir=synthetic_case_dir,
                source=synthetic_source,
                output_dir=output_dir,
                settings=ExportSettings(),
            )

        assert result == 0
        mapcut = read_mapcut(output_dir / "mapcut.rv0")
        assert mapcut.codigos_uhes == (101, 102, 103)
        assert mapcut.codigos_uhes_jusante == (0, 1, 1)
        assert mapcut.lag_meses_gnl == (3,)

    def test_plant_records_follow_slot_order_not_id_order(
        self,
        synthetic_case_dir: Path,
        synthetic_source: SourceDeck,
        mock_manifest: PolicyManifest,
        tmp_path: Path,
    ) -> None:
        # novomodelo orders plants by start date before id, so slots can run 3, 1, 2.
        slots = _make_slots(3, entity_ids=(3, 1, 2))
        pools = {i: _make_pool(i, i, i, slots, n_pieces=5) for i in range(3)}
        output_dir = tmp_path / "dessem_fcf"

        with patch(
            "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
        ) as read:
            read.return_value = PolicyCheckpoint(mock_manifest, pools)
            result = run_export(
                case_dir=synthetic_case_dir,
                source=synthetic_source,
                output_dir=output_dir,
                settings=ExportSettings(),
            )

        assert result == 0
        mapcut = read_mapcut(output_dir / "mapcut.rv0")
        assert mapcut.codigos_uhes == (103, 101, 102)
        # UHE_C and UHE_B drain into UHE_A, now the second plant.
        assert mapcut.codigos_uhes_jusante == (2, 0, 2)

    @pytest.mark.parametrize(
        ("hydros", "match"),
        [
            pytest.param(
                {
                    1: SourceHydro(code=101, name="UHE_A"),
                    2: SourceHydro(code=102, name="UHE_B"),
                },
                "the deck operates 2 hydro plants but the case holds 3",
                id="plant-set",
            ),
            pytest.param(
                {
                    1: SourceHydro(code=101, name="UHE_A"),
                    2: SourceHydro(code=102, name="UHE_X"),
                    3: SourceHydro(code=103, name="UHE_C"),
                },
                "1 hydro plant names differ",
                id="plant-name",
            ),
        ],
    )
    def test_deck_other_than_the_source_fails_the_export(
        self,
        synthetic_case_dir: Path,
        synthetic_source: SourceDeck,
        mock_manifest: PolicyManifest,
        mock_pools: dict[int, StageCutPool],
        tmp_path: Path,
        hydros: dict[int, SourceHydro],
        match: str,
    ) -> None:
        output_dir = tmp_path / "dessem_fcf"
        wrong_deck = replace(synthetic_source, hydros=hydros)

        with (
            dx.collect() as diagnostics,
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as read,
        ):
            read.return_value = PolicyCheckpoint(mock_manifest, mock_pools)
            result = run_export(
                case_dir=synthetic_case_dir,
                source=wrong_deck,
                output_dir=output_dir,
                settings=ExportSettings(),
            )

        assert result == 1
        assert not (output_dir / "mapcut.rv0").exists()
        errors = [d for d in diagnostics if d.code == "dessem-fcf-export-failed"]
        assert len(errors) == 1
        assert match in errors[0].summary

    def test_manifest_contains_metadata(
        self,
        synthetic_case_dir: Path,
        synthetic_source: SourceDeck,
        mock_manifest: PolicyManifest,
        mock_pools: dict[int, StageCutPool],
        tmp_path: Path,
    ) -> None:
        """Test that export manifest contains required metadata."""
        output_dir = tmp_path / "dessem_fcf"

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as mock_read,
        ):
            mock_read.return_value = PolicyCheckpoint(mock_manifest, mock_pools)

            settings = ExportSettings(
                eco_subdirectory="eco",
                content_subdirectory="content",
                include_terminal_pool=False,
            )

            run_export(
                case_dir=synthetic_case_dir,
                source=synthetic_source,
                output_dir=output_dir,
                settings=settings,
            )

        manifest_path = output_dir / "export_manifest.json"
        manifest: dict[str, Any] = json.loads(manifest_path.read_text())

        assert manifest["status"] == "ok"
        assert manifest["revision"] == "rv0"
        assert manifest["source_path"] == str(synthetic_source.directory.resolve())
        assert "premises" in manifest
        assert len(manifest["premises"]) == 15
        assert "outputs" in manifest
        assert "mapcut" in manifest["outputs"]
        assert "cortdeco" in manifest["outputs"]

    def test_eco_csvs_created(
        self,
        synthetic_case_dir: Path,
        synthetic_source: SourceDeck,
        mock_manifest: PolicyManifest,
        mock_pools: dict[int, StageCutPool],
        tmp_path: Path,
    ) -> None:
        """Test that ECO CSVs are created when emit_audit_csv is True."""
        output_dir = tmp_path / "dessem_fcf"

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as mock_read,
        ):
            mock_read.return_value = PolicyCheckpoint(mock_manifest, mock_pools)

            settings = ExportSettings(
                eco_subdirectory="eco",
                content_subdirectory="content",
                include_terminal_pool=False,
                emit_audit_csv=True,
            )

            run_export(
                case_dir=synthetic_case_dir,
                source=synthetic_source,
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
        synthetic_source: SourceDeck,
        mock_manifest: PolicyManifest,
        mock_pools: dict[int, StageCutPool],
        tmp_path: Path,
    ) -> None:
        """Test that content CSVs are created when emit_audit_csv is True."""
        output_dir = tmp_path / "dessem_fcf"

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as mock_read,
        ):
            mock_read.return_value = PolicyCheckpoint(mock_manifest, mock_pools)

            settings = ExportSettings(
                eco_subdirectory="eco",
                content_subdirectory="content",
                include_terminal_pool=False,
                emit_audit_csv=True,
            )

            run_export(
                case_dir=synthetic_case_dir,
                source=synthetic_source,
                output_dir=output_dir,
                settings=settings,
            )

        content_dir = output_dir / "content"
        assert content_dir.exists()
        assert (content_dir / "mapcut_content.csv").exists()
        assert (content_dir / "cortdeco_content.csv").exists()


def _bucket(start: int, end: int) -> EntitySlotRecord:
    return EntitySlotRecord(
        entity_type=3,
        entity_id=1,
        subindex=0,
        was_active=True,
        reference_date=UNDATED,
        interval_start=start,
        interval_end=end,
    )


class TestTravelTimeExport:
    """UHE_B's water reaches UHE_A 240 h after release, on weekly stages."""

    @pytest.fixture
    def exported(
        self, travel_time_case_dir: Path, synthetic_source: SourceDeck, tmp_path: Path
    ) -> tuple[Path, list[str]]:
        # Each pool's two buckets are the two weeks after its stage ends.
        bucket_dates = (20260108, 20260115, 20260122, 20260129, 20260205)
        pools = {
            stage: _make_pool(
                stage,
                stage,
                stage,
                _make_slots(3)
                + (
                    _bucket(bucket_dates[stage], bucket_dates[stage + 1]),
                    _bucket(bucket_dates[stage + 1], bucket_dates[stage + 2]),
                ),
                n_pieces=5,
            )
            for stage in range(3)
        }
        output_dir = tmp_path / "dessem_fcf"
        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as read,
            dx.collect() as diagnostics,
        ):
            read.return_value = PolicyCheckpoint(_make_manifest(), pools)
            result = run_export(
                case_dir=travel_time_case_dir,
                source=replace(synthetic_source, travel_time_codes=(102,)),
                output_dir=output_dir,
                settings=ExportSettings(emit_audit_csv=True),
            )

        assert result == 0
        return output_dir, [diagnostic.code for diagnostic in diagnostics]

    def test_mapcut_declares_the_plant_and_its_stage_lags(
        self, exported: tuple[Path, list[str]]
    ) -> None:
        output_dir, codes = exported
        mapcut, _ = cross_check_pair(
            output_dir / "mapcut.rv0", output_dir / "cortdeco.rv0"
        )

        assert mapcut.codigos_uhes_tempo_viagem == (102,)
        assert mapcut.travel_time_records[0].numero_horas == 240
        assert mapcut.travel_time_lags == (2, 2, 2)
        assert "dessem-fcf-travel-time-exported" in codes

    def test_each_release_takes_the_buckets_it_arrives_in(
        self, exported: tuple[Path, list[str]]
    ) -> None:
        # The stage's release arrives 96 h into the first bucket and 72 h into
        # the second; the release before it has 72 h left in the first.
        output_dir, _ = exported
        _, cortdeco = cross_check_pair(
            output_dir / "mapcut.rv0", output_dir / "cortdeco.rv0"
        )

        for cut in cortdeco.cuts:
            piece = round(cut.rhs) - 1
            first, second = 103.0 + piece, 104.0 + piece
            np.testing.assert_allclose(
                cut.pi_qdefp,
                [(96 * first + 72 * second) / 168 / 1000, 72 * first / 168 / 1000],
            )

    def test_content_csvs_mirror_the_travel_time_records(
        self, exported: tuple[Path, list[str]]
    ) -> None:
        output_dir, _ = exported
        content = output_dir / "content"
        cortdeco_header = (content / "cortdeco_content.csv").read_text().splitlines()[0]
        mapcut_rows = (content / "mapcut_content.csv").read_text().splitlines()

        assert "pi_qdefp_uhe102_lag0,pi_qdefp_uhe102_lag1,pi_gnl" in cortdeco_header
        assert "20,reg7_travel_time,1,codigo_usina,0,102" in mapcut_rows
        assert "20,reg7_travel_time,1,numero_horas,1,240" in mapcut_rows
        assert "21,reg8_travel_time,1,indice_lag,0,1" in mapcut_rows


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

    def test_failure_manifest_on_format_error(
        self,
        synthetic_case_dir: Path,
        synthetic_source: SourceDeck,
        tmp_path: Path,
    ) -> None:
        """Test that failure manifest is written on PolicyFormatError."""
        from novomodelo_bridge.dessem.exporters.fcf.core.reader import PolicyFormatError

        output_dir = tmp_path / "dessem_fcf"

        with patch(
            "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
        ) as mock_read:
            mock_read.side_effect = PolicyFormatError("test error")

            settings = ExportSettings(
                eco_subdirectory="eco",
                content_subdirectory="content",
                include_terminal_pool=False,
            )

            result = run_export(
                case_dir=synthetic_case_dir,
                source=synthetic_source,
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
        synthetic_source: SourceDeck,
        tmp_path: Path,
    ) -> None:
        """Test that failure manifest is written on MappingError."""
        from novomodelo_bridge.dessem.exporters.fcf.core.mapper import MappingError

        output_dir = tmp_path / "dessem_fcf"
        mock_manifest = _make_manifest()
        mock_pools = {
            i: _make_pool(i, i, i, _make_slots(3), n_pieces=5) for i in range(4)
        }

        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as mock_read,
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.travel_time_plants"
            ) as mock_travel,
        ):
            mock_read.return_value = PolicyCheckpoint(mock_manifest, mock_pools)
            mock_travel.side_effect = MappingError("plant 102 has no VI record")

            settings = ExportSettings(
                eco_subdirectory="eco",
                content_subdirectory="content",
                include_terminal_pool=False,
            )

            result = run_export(
                case_dir=synthetic_case_dir,
                source=synthetic_source,
                output_dir=output_dir,
                settings=settings,
            )

        assert result == 1
        manifest_path = output_dir / "export_manifest.json"
        assert manifest_path.exists()

        manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        assert manifest["status"] == "failed"
