"""Integration tests for dessem/exporters/fcf/pipeline.py."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.dessem.exporters.fcf.core.cortdeco_writer import (
    node_and_iteration,
)
from novomodelo_bridge.dessem.exporters.fcf.core.cross_check import (
    CortdecoContents,
    cross_check_pair,
    read_mapcut,
    reject_pair,
)
from novomodelo_bridge.dessem.exporters.fcf.core.layout import (
    RECORD_SIZE,
    TAMANHO_CORTE,
    LayoutError,
)
from novomodelo_bridge.dessem.exporters.fcf.core.mapper import (
    ENTITY_TYPE_ANTICIPATED_THERMAL,
    SourceHydro,
)
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
from tests.dessem.conftest import weekly_windows, write_commitments


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


def _anticipated(
    thermal_id: int, subindex: int, start: int, end: int
) -> EntitySlotRecord:
    return EntitySlotRecord(
        entity_type=ENTITY_TYPE_ANTICIPATED_THERMAL,
        entity_id=thermal_id,
        subindex=subindex,
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


_DELIVERY_MISSING = "dessem-fcf-gnl-delivery-missing"
_NO_SLOT = "no slot in the policy"
_SHORT_LEAD = "lead time shorter than the delivery stage"


def _rewrite_gnl_lead(case_dir: Path, lead_time_hours: float | None) -> None:
    """Set GNL_A's lead time in `thermals.json`, or make it a non-GNL thermal without commitment windows."""
    path = case_dir / "system" / "thermals.json"
    thermals = json.loads(path.read_text())
    gnl_a = next(t for t in thermals["thermals"] if t["name"] == "GNL_A")
    if lead_time_hours is None:
        del gnl_a["anticipated_config"]
        conditions_path = case_dir / "initial_conditions.json"
        conditions = json.loads(conditions_path.read_text())
        conditions["past_anticipated_commitments"] = [
            window
            for window in conditions["past_anticipated_commitments"]
            if window["thermal_id"] != gnl_a["id"]
        ]
        conditions_path.write_text(json.dumps(conditions))
    else:
        gnl_a["anticipated_config"]["lead_time_hours"] = lead_time_hours
    path.write_text(json.dumps(thermals))


def _keep_first_stage(case_dir: Path) -> None:
    path = case_dir / "stages.json"
    stages = json.loads(path.read_text())
    stages["stages"] = stages["stages"][:1]
    path.write_text(json.dumps(stages))


class TestGnlDeliverySelection:
    """Each GNL plant's slot for the second stage's delivery, found before any write."""

    @staticmethod
    def _pools(*slots: EntitySlotRecord) -> dict[int, StageCutPool]:
        return {
            stage: _make_pool(stage, stage, stage, _make_slots(3) + slots, n_pieces=5)
            for stage in range(3)
        }

    @staticmethod
    def _export(
        case_dir: Path,
        source: SourceDeck,
        output_dir: Path,
        pools: dict[int, StageCutPool],
        manifest: PolicyManifest | None = None,
    ) -> tuple[int, list[dx.Diagnostic]]:
        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as read,
            dx.collect() as diagnostics,
        ):
            read.return_value = PolicyCheckpoint(manifest or _make_manifest(), pools)
            result = run_export(
                case_dir=case_dir,
                source=source,
                output_dir=output_dir,
                settings=ExportSettings(),
            )
        return result, diagnostics

    @staticmethod
    def _missing(diagnostics: list[dx.Diagnostic]) -> list[dx.Diagnostic]:
        return [d for d in diagnostics if d.code == _DELIVERY_MISSING]

    def test_plant_without_a_slot_is_listed_with_the_no_slot_cause(
        self, synthetic_case_dir: Path, synthetic_source: SourceDeck, tmp_path: Path
    ) -> None:
        result, diagnostics = self._export(
            synthetic_case_dir, synthetic_source, tmp_path, _make_pools_dict()
        )

        assert result == 0
        (missing,) = self._missing(diagnostics)
        assert missing.severity == dx.Severity.WARNING
        assert missing.table is not None
        assert missing.table.rows == [[86, "GNL_A", "2026-01-08", 1512.0, _NO_SLOT]]
        assert len(missing.notes) == 1

    @pytest.mark.parametrize(
        ("lead_time_hours", "cause"),
        [
            pytest.param(24.0, _SHORT_LEAD, id="shorter-than-the-stage"),
            pytest.param(168.0, _NO_SLOT, id="equal-to-the-stage"),
        ],
    )
    def test_lead_time_against_the_delivery_stage_decides_the_cause(
        self,
        synthetic_case_dir: Path,
        synthetic_source: SourceDeck,
        tmp_path: Path,
        lead_time_hours: float,
        cause: str,
    ) -> None:
        _rewrite_gnl_lead(synthetic_case_dir, lead_time_hours)

        result, diagnostics = self._export(
            synthetic_case_dir, synthetic_source, tmp_path, _make_pools_dict()
        )

        assert result == 0
        (missing,) = self._missing(diagnostics)
        assert missing.severity == dx.Severity.WARNING
        assert missing.table is not None
        assert missing.table.rows == [
            [86, "GNL_A", "2026-01-08", lead_time_hours, cause]
        ]
        assert len(missing.notes) == 1

    def test_note_follows_the_cause(
        self, synthetic_case_dir: Path, synthetic_source: SourceDeck, tmp_path: Path
    ) -> None:
        notes = {}
        for lead_time_hours in (24.0, 1512.0):
            _rewrite_gnl_lead(synthetic_case_dir, lead_time_hours)
            _, diagnostics = self._export(
                synthetic_case_dir,
                synthetic_source,
                tmp_path / str(lead_time_hours),
                _make_pools_dict(),
            )
            (missing,) = self._missing(diagnostics)
            notes[lead_time_hours] = missing.notes

        assert notes[24.0] != notes[1512.0]

    def test_every_plant_with_its_slot_emits_no_warning(
        self, gnl_case_dir: Path, gnl_source: SourceDeck, tmp_path: Path
    ) -> None:
        pools = self._pools(
            _anticipated(100, 0, 20260108, 20260115),
            _anticipated(102, 0, 20260108, 20260115),
        )

        result, diagnostics = self._export(gnl_case_dir, gnl_source, tmp_path, pools)

        assert result == 0
        assert self._missing(diagnostics) == []

    @pytest.mark.parametrize(
        "slots",
        [
            pytest.param(
                (
                    _anticipated(100, 0, 20260108, 20260116),
                    _anticipated(102, 0, 20260108, 20260115),
                ),
                id="hours-differ-from-the-blocks",
            ),
            pytest.param(
                (
                    _anticipated(100, 0, 20260108, 20260115),
                    _anticipated(100, 1, 20260108, 20260115),
                    _anticipated(102, 0, 20260108, 20260115),
                ),
                id="two-slots-for-one-delivery",
            ),
        ],
    )
    def test_ambiguous_delivery_fails_the_export_before_any_write(
        self,
        gnl_case_dir: Path,
        gnl_source: SourceDeck,
        tmp_path: Path,
        slots: tuple[EntitySlotRecord, ...],
    ) -> None:
        result, diagnostics = self._export(
            gnl_case_dir, gnl_source, tmp_path, self._pools(*slots)
        )

        assert result == 1
        assert [d.code for d in diagnostics].count("dessem-fcf-export-failed") == 1
        manifest = json.loads((tmp_path / "export_manifest.json").read_text())
        assert manifest["status"] == "failed"
        assert manifest["failed_step"] == "validate premises"
        assert not (tmp_path / "mapcut.rv0").exists()
        assert not (tmp_path / "cortdeco.rv0").exists()

    def test_single_stage_case_with_gnl_fails_the_export(
        self, synthetic_case_dir: Path, synthetic_source: SourceDeck, tmp_path: Path
    ) -> None:
        _keep_first_stage(synthetic_case_dir)

        result, diagnostics = self._export(
            synthetic_case_dir,
            synthetic_source,
            tmp_path,
            {0: _make_pool(0, 0, 0, _make_slots(3), n_pieces=5)},
            _make_manifest(n_stages=1, n_pools=2),
        )

        assert result == 1
        assert "dessem-fcf-export-failed" in [d.code for d in diagnostics]

    def test_single_stage_case_without_gnl_still_exports(
        self, synthetic_case_dir: Path, synthetic_source: SourceDeck, tmp_path: Path
    ) -> None:
        _keep_first_stage(synthetic_case_dir)
        _rewrite_gnl_lead(synthetic_case_dir, None)

        result, diagnostics = self._export(
            synthetic_case_dir,
            replace(synthetic_source, gnl_thermals=()),
            tmp_path,
            {0: _make_pool(0, 0, 0, _make_slots(3), n_pieces=5)},
            _make_manifest(n_stages=1, n_pools=2),
        )

        assert result == 0
        assert self._missing(diagnostics) == []


_MAPPER_LOGGER = "novomodelo_bridge.dessem.exporters.fcf.core.mapper"

# Positions 3-6 on weekly stages: thermal 100's delivery from 2026-01-08 is
# position 3 and thermal 102's is position 6; positions 4 and 5 are decoys.
_GNL_SLOTS = _make_slots(3) + (
    _anticipated(100, 0, 20260108, 20260115),
    _anticipated(100, 1, 20260115, 20260122),
    _anticipated(102, 0, 20260115, 20260122),
    _anticipated(102, 2, 20260108, 20260115),
)


def _gnl_coefficients(piece: int) -> NDArray[np.float64]:
    """Piece *piece*: storage, then -(200 + piece) and -(150 + piece) R$/MWh x 168 h."""
    coefficients = np.array(
        [100.0 + piece + j for j in range(3)]
        + [-33600.0 - 168 * piece, -50400.0, -8400.0, -25200.0 - 168 * piece]
    )
    coefficients.setflags(write=False)
    return coefficients


def _gnl_pool(
    pool_id: int,
    slots: tuple[EntitySlotRecord, ...],
    coefficients: Sequence[NDArray[np.float64]],
) -> StageCutPool:
    """A pool at stage *pool_id* with one piece per coefficient vector."""
    pieces = tuple(
        AffinePieceRecord(
            slot_index=k,
            iteration=k + 1,
            forward_pass_index=k,
            intercept=1000.0 * (k + 1),
            coefficients=vector,
            is_active=True,
        )
        for k, vector in enumerate(coefficients)
    )
    pool = _make_pool(pool_id, pool_id, pool_id, slots, n_pieces=len(pieces))
    return replace(pool, pieces=pieces)


class TestGnlExport:
    """Only node 0's cuts carry GNL, in R$/MWh, at the second stage's delivery."""

    @pytest.fixture
    def exported(
        self,
        gnl_case_dir: Path,
        gnl_source: SourceDeck,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> tuple[CortdecoContents, list[str]]:
        pools = {
            pool_id: _gnl_pool(
                pool_id, _GNL_SLOTS, [_gnl_coefficients(k) for k in range(5)]
            )
            for pool_id in range(3)
        }
        output_dir = tmp_path / "dessem_fcf"
        with (
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as read,
            caplog.at_level(logging.WARNING, logger=_MAPPER_LOGGER),
        ):
            read.return_value = PolicyCheckpoint(_make_manifest(), pools)
            result = run_export(
                case_dir=gnl_case_dir,
                source=gnl_source,
                output_dir=output_dir,
                settings=ExportSettings(),
            )

        assert result == 0
        _, cortdeco = cross_check_pair(
            output_dir / "mapcut.rv0", output_dir / "cortdeco.rv0"
        )
        drops = [
            message
            for record in caplog.records
            if (message := record.getMessage()).startswith("premise P14 drops")
        ]
        return cortdeco, drops

    def test_first_node_cuts_carry_the_delivery_coefficients_in_r_per_mwh(
        self, exported: tuple[CortdecoContents, list[str]]
    ) -> None:
        cortdeco, _ = exported

        first_node = [
            cut
            for cut in cortdeco.cuts
            if not cut.is_extra_record
            and node_and_iteration(cut.record_index, 3)[0] == 0
        ]

        assert len(first_node) == 5
        for cut in first_node:
            piece = round(cut.rhs) - 1
            expected = np.zeros(18)
            expected[0:3] = 200 + piece
            expected[9:12] = 150 + piece
            assert cut.pi_gnl.tolist() == expected.tolist()

    def test_later_nodes_and_the_extra_record_carry_zeros(
        self, exported: tuple[CortdecoContents, list[str]]
    ) -> None:
        cortdeco, _ = exported

        later = [
            cut
            for cut in cortdeco.cuts
            if cut.is_extra_record or node_and_iteration(cut.record_index, 3)[0] > 0
        ]

        assert len(later) == 11
        assert all(cut.pi_gnl.tolist() == [0.0] * 18 for cut in later)

    def test_decoy_slots_never_reach_the_file(
        self, exported: tuple[CortdecoContents, list[str]]
    ) -> None:
        cortdeco, _ = exported

        written = np.concatenate([cut.pi_gnl for cut in cortdeco.cuts])

        assert not np.isin([300.0, 50.0], written).any()

    def test_drop_count_is_logged_once_with_its_per_pool_breakdown(
        self, exported: tuple[CortdecoContents, list[str]]
    ) -> None:
        _, drops = exported

        assert drops == [
            "premise P14 drops 50 anticipated-thermal coefficients: "
            "pool 0=10, pool 1=20, pool 2=20"
        ]


_EXPORTED = "dessem-fcf-gnl-coefficients-exported"
_ZERO = "dessem-fcf-gnl-coefficient-zero"
_REPEATED = "dessem-fcf-gnl-submarket-repeated"
_ZEROED = "dessem-fcf-gnl-coefficients-zeroed"
_COMMITTED = "dessem-fcf-gnl-committed-cost-exported"
_GNL_A_ROW = [86, "GNL_A", 1, "2026-01-08 to 2026-01-15", 168.0, 200.0, 204.0]
_GNL_B_ROW = [87, "GNL_B", 3, "2026-01-08 to 2026-01-15", 168.0, 150.0, 154.0]
_REPOSITORY_TOKENS = (
    "docs/",
    "plans/",
    "~/git",
    "ticket-",
    "epic-",
    "src/",
    ".py",
    "TRACKED",
    "C24",
)

# Thermal 102's two slots, positions 5 and 6 of `_GNL_SLOTS`, lose their dates.
_UNDATED_GNL_B_SLOTS = _GNL_SLOTS[:5] + tuple(
    replace(slot, interval_start=UNDATED, interval_end=UNDATED)
    for slot in _GNL_SLOTS[5:]
)


@dataclass(frozen=True)
class _GnlScenario:
    """One export of `TestGnlExport`'s inputs, with GNL_B and its policy edited."""

    slots: tuple[EntitySlotRecord, ...] = _GNL_SLOTS
    zero_gnl_b_in_first_cut: bool = False
    gnl_b_bus: int = 2
    gnl_b_lead_time_hours: float = 1512.0


def _edit_gnl_b(case_dir: Path, *, bus_id: int, lead_time_hours: float) -> None:
    path = case_dir / "system" / "thermals.json"
    thermals = json.loads(path.read_text())
    gnl_b = next(t for t in thermals["thermals"] if t["name"] == "GNL_B")
    gnl_b["bus_id"] = bus_id
    gnl_b["anticipated_config"]["lead_time_hours"] = lead_time_hours
    path.write_text(json.dumps(thermals))


def _run_export_on_policy(
    case_dir: Path,
    source: SourceDeck,
    output_dir: Path,
    pools: dict[int, StageCutPool],
    manifest: PolicyManifest,
) -> tuple[int, list[dx.Diagnostic]]:
    with (
        patch("novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy") as read,
        dx.collect() as diagnostics,
    ):
        read.return_value = PolicyCheckpoint(manifest, pools)
        result = run_export(
            case_dir=case_dir,
            source=source,
            output_dir=output_dir,
            settings=ExportSettings(),
        )
    return result, diagnostics


def _export_gnl(
    case_dir: Path, source: SourceDeck, output_dir: Path, scenario: _GnlScenario
) -> tuple[int, list[dx.Diagnostic]]:
    _edit_gnl_b(
        case_dir,
        bus_id=scenario.gnl_b_bus,
        lead_time_hours=scenario.gnl_b_lead_time_hours,
    )
    vectors = [_gnl_coefficients(k) for k in range(5)]
    if scenario.zero_gnl_b_in_first_cut:
        zeroed = vectors[0].copy()
        zeroed[6] = 0.0
        zeroed.setflags(write=False)
        vectors[0] = zeroed
    pools = {
        pool_id: _gnl_pool(pool_id, scenario.slots, vectors) for pool_id in range(3)
    }
    return _run_export_on_policy(case_dir, source, output_dir, pools, _make_manifest())


def _with_code(diagnostics: list[dx.Diagnostic], code: str) -> list[dx.Diagnostic]:
    return [d for d in diagnostics if d.code == code]


class TestGnlDiagnostics:
    """The GNL export reports what the file holds, read back from the validated pair."""

    def test_exported_rows_are_read_back_from_the_file_with_the_drop_note(
        self, gnl_case_dir: Path, gnl_source: SourceDeck, tmp_path: Path
    ) -> None:
        result, diagnostics = _export_gnl(
            gnl_case_dir, gnl_source, tmp_path, _GnlScenario()
        )

        assert result == 0
        (exported,) = _with_code(diagnostics, _EXPORTED)
        assert exported.severity == dx.Severity.INFO
        assert exported.category == "FCF export"
        assert exported.table is not None
        assert exported.table.columns == [
            "GNL plant",
            "Name",
            "Submarket",
            "Delivery",
            "Hours",
            "Min R$/MWh",
            "Max R$/MWh",
        ]
        assert exported.table.rows == [_GNL_A_ROW, _GNL_B_ROW]
        drop_notes = [
            note
            for note in exported.notes
            if note.startswith("50 ") and "intercept" in note
        ]
        assert len(drop_notes) == 1
        assert {d.code for d in diagnostics}.isdisjoint({_ZERO, _REPEATED, _ZEROED})

    def test_no_drop_note_when_nothing_is_dropped(
        self, gnl_case_dir: Path, gnl_source: SourceDeck, tmp_path: Path
    ) -> None:
        delivered = _GNL_SLOTS[:3] + (_GNL_SLOTS[3], _GNL_SLOTS[6])
        pools = {
            0: _gnl_pool(
                0,
                delivered,
                [_gnl_coefficients(k)[[0, 1, 2, 3, 6]] for k in range(5)],
            ),
            **{
                pool_id: _gnl_pool(
                    pool_id,
                    _make_slots(3),
                    [_gnl_coefficients(k)[:3] for k in range(5)],
                )
                for pool_id in (1, 2)
            },
        }
        result, diagnostics = _run_export_on_policy(
            gnl_case_dir, gnl_source, tmp_path, pools, _make_manifest()
        )

        assert result == 0
        (exported,) = _with_code(diagnostics, _EXPORTED)
        assert exported.table is not None
        assert exported.table.rows == [_GNL_A_ROW, _GNL_B_ROW]
        assert not any("not written" in note for note in exported.notes)

    def test_a_zero_in_one_first_node_cut_is_counted_per_plant(
        self, gnl_case_dir: Path, gnl_source: SourceDeck, tmp_path: Path
    ) -> None:
        result, diagnostics = _export_gnl(
            gnl_case_dir,
            gnl_source,
            tmp_path,
            _GnlScenario(zero_gnl_b_in_first_cut=True),
        )

        assert result == 0
        (zero,) = _with_code(diagnostics, _ZERO)
        assert zero.severity == dx.Severity.WARNING
        assert zero.table is not None
        assert zero.table.columns == ["GNL plant", "Name", "Zero cuts", "Cuts"]
        assert zero.table.rows == [[87, "GNL_B", 1, 5]]
        (exported,) = _with_code(diagnostics, _EXPORTED)
        assert exported.table is not None
        assert exported.table.rows[1][5:] == [0.0, 154.0]

    def test_the_extra_record_is_not_counted_with_one_cut_building_node(
        self, gnl_case_dir: Path, gnl_source: SourceDeck, tmp_path: Path
    ) -> None:
        stages_path = gnl_case_dir / "stages.json"
        stages = json.loads(stages_path.read_text())
        stages["stages"] = stages["stages"][:2]
        stages_path.write_text(json.dumps(stages))
        vectors = [_gnl_coefficients(k) for k in range(5)]
        last_cut = vectors[4].copy()
        last_cut[6] = 0.0
        pools = {0: _gnl_pool(0, _GNL_SLOTS, [*vectors[:4], last_cut])}

        result, diagnostics = _run_export_on_policy(
            gnl_case_dir,
            gnl_source,
            tmp_path,
            pools,
            _make_manifest(n_stages=2, n_pools=2),
        )

        assert result == 0
        (zero,) = _with_code(diagnostics, _ZERO)
        assert zero.table is not None
        assert zero.table.rows == [[87, "GNL_B", 1, 5]]

    def test_plants_sharing_a_submarket_are_listed_once(
        self, gnl_case_dir: Path, gnl_source: SourceDeck, tmp_path: Path
    ) -> None:
        result, diagnostics = _export_gnl(
            gnl_case_dir, gnl_source, tmp_path, _GnlScenario(gnl_b_bus=0)
        )

        assert result == 0
        (repeated,) = _with_code(diagnostics, _REPEATED)
        assert repeated.severity == dx.Severity.WARNING
        assert repeated.table is not None
        assert repeated.table.columns == ["Submarket", "GNL plants"]
        assert repeated.table.rows == [[1, "86, 87"]]

    @pytest.mark.parametrize(
        ("lead_time_hours", "missing_row"),
        [
            pytest.param(
                1512.0, [87, "GNL_B", "2026-01-08", 1512.0, _NO_SLOT], id="no-slot"
            ),
            pytest.param(
                24.0, [87, "GNL_B", "2026-01-08", 24.0, _SHORT_LEAD], id="short-lead"
            ),
        ],
    )
    def test_a_plant_without_a_delivery_is_missing_not_zero(
        self,
        gnl_case_dir: Path,
        gnl_source: SourceDeck,
        tmp_path: Path,
        lead_time_hours: float,
        missing_row: list[object],
    ) -> None:
        result, diagnostics = _export_gnl(
            gnl_case_dir,
            gnl_source,
            tmp_path,
            _GnlScenario(
                slots=_UNDATED_GNL_B_SLOTS, gnl_b_lead_time_hours=lead_time_hours
            ),
        )

        assert result == 0
        (exported,) = _with_code(diagnostics, _EXPORTED)
        assert exported.table is not None
        assert exported.table.rows == [_GNL_A_ROW]
        (missing,) = _with_code(diagnostics, _DELIVERY_MISSING)
        assert missing.table is not None
        assert missing.table.rows == [missing_row]
        assert _with_code(diagnostics, _ZERO) == []

    @pytest.mark.parametrize(
        ("scenario", "code"),
        [
            pytest.param(_GnlScenario(), _EXPORTED, id="exported"),
            pytest.param(_GnlScenario(zero_gnl_b_in_first_cut=True), _ZERO, id="zero"),
            pytest.param(_GnlScenario(gnl_b_bus=0), _REPEATED, id="repeated"),
            pytest.param(
                _GnlScenario(slots=_UNDATED_GNL_B_SLOTS),
                _DELIVERY_MISSING,
                id="no-slot",
            ),
            pytest.param(
                _GnlScenario(slots=_UNDATED_GNL_B_SLOTS, gnl_b_lead_time_hours=24.0),
                _DELIVERY_MISSING,
                id="short-lead",
            ),
            pytest.param(_GnlScenario(), _COMMITTED, id="committed-cost"),
        ],
    )
    def test_gnl_diagnostics_hold_no_repository_internal_reference(
        self,
        gnl_case_dir: Path,
        gnl_source: SourceDeck,
        tmp_path: Path,
        scenario: _GnlScenario,
        code: str,
    ) -> None:
        _, diagnostics = _export_gnl(gnl_case_dir, gnl_source, tmp_path, scenario)

        scanned = [d for d in diagnostics if d.code.startswith("dessem-fcf-gnl-")]
        assert code in {d.code for d in scanned}
        for diagnostic in scanned:
            text = json.dumps(diagnostic.to_dict())
            assert [t for t in _REPOSITORY_TOKENS if t in text] == []


class TestGnlMapcutHours:
    """Reg 9 carries the delivery stage's block hours, not the first stage's."""

    @pytest.fixture
    def exported_pair(
        self, gnl_case_dir: Path, gnl_source: SourceDeck, tmp_path: Path
    ) -> tuple[Path, Path]:
        output_dir = tmp_path / "dessem_fcf"
        result, _ = _export_gnl(gnl_case_dir, gnl_source, output_dir, _GnlScenario())
        assert result == 0
        return output_dir / "mapcut.rv0", output_dir / "cortdeco.rv0"

    def test_reg9_trailing_block_is_the_delivery_stages_hours_patamar_major(
        self, exported_pair: tuple[Path, Path]
    ) -> None:
        mapcut, _ = cross_check_pair(*exported_pair)

        assert mapcut.gnl_trailing_block == (27.0, 27.0, 68.0, 68.0, 73.0, 73.0)

    def test_cross_check_refuses_a_reg9_patamar_count_that_disagrees_with_reg6(
        self, exported_pair: tuple[Path, Path]
    ) -> None:
        mapcut_path, cortdeco_path = exported_pair
        contents = read_mapcut(mapcut_path)
        ngnl = len(contents.codigos_submercados_gnl)
        with mapcut_path.open("r+b") as handle:
            handle.seek(contents.gnl_record_indices[0] * RECORD_SIZE + 4 + 8 * ngnl)
            handle.write(np.array([2], dtype="<i4").tobytes())

        with pytest.raises(LayoutError, match=r"patamares_gnl=\(2, 3\).*3 patamares"):
            cross_check_pair(mapcut_path, cortdeco_path)


class TestGnlCommittedCostExport:
    """Reg 10's last value is the cost of the GNL generation committed before the study."""

    def test_each_stage_carries_its_committed_cost_and_nothing_else(
        self, gnl_case_dir: Path, gnl_source: SourceDeck, tmp_path: Path
    ) -> None:
        write_commitments(
            gnl_case_dir,
            weekly_windows(100, (10.0, 0.0, 0.0))
            + weekly_windows(102, (5.0, 20.0, 0.0)),
        )
        output_dir = tmp_path / "dessem_fcf"

        result, _ = _export_gnl(gnl_case_dir, gnl_source, output_dir, _GnlScenario())

        assert result == 0
        records = read_mapcut(output_dir / "mapcut.rv0").cost_records
        assert [record[5] for record in records] == [588.0, 1008.0, 0.0]
        assert all(record[1:5] == (0.0, 0.0, 0.0, 0.0) for record in records)


class TestGnlCommittedCostDiagnostic:
    """The committed-cost diagnostic reads its value back from mapcut and keeps a fixed header."""

    _COLUMNS = ["Stage", "Period", "Hours", "GNL plants", "10³ R$"]

    @staticmethod
    def _export_two_plants(
        case_dir: Path, source: SourceDeck, output_dir: Path
    ) -> list[dx.Diagnostic]:
        write_commitments(
            case_dir,
            weekly_windows(100, (10.0, 0.0, 0.0)) + weekly_windows(102, (5.0, 20.0)),
        )
        result, diagnostics = _export_gnl(case_dir, source, output_dir, _GnlScenario())
        assert result == 0
        return _with_code(diagnostics, _COMMITTED)

    @staticmethod
    def _export_zero_mw(
        case_dir: Path, source: SourceDeck, output_dir: Path
    ) -> list[dx.Diagnostic]:
        result, diagnostics = _run_export_on_policy(
            case_dir, source, output_dir, _make_pools_dict(), _make_manifest()
        )
        assert result == 0
        return _with_code(diagnostics, _COMMITTED)

    def test_two_plant_table_lists_every_plant_in_every_stage(
        self, gnl_case_dir: Path, gnl_source: SourceDeck, tmp_path: Path
    ) -> None:
        (committed,) = self._export_two_plants(gnl_case_dir, gnl_source, tmp_path)

        assert committed.severity == dx.Severity.INFO
        assert committed.category == "FCF export"
        assert committed.table is not None
        assert committed.table.columns == self._COLUMNS
        assert committed.table.rows == [
            [
                0,
                "2026-01-01 to 2026-01-08",
                168.0,
                "86 GNL_A: 10 MW, 87 GNL_B: 5 MW",
                588.0,
            ],
            [
                1,
                "2026-01-08 to 2026-01-15",
                168.0,
                "86 GNL_A: 0 MW, 87 GNL_B: 20 MW",
                1008.0,
            ],
            [2, "2026-01-15 to 2026-01-22", 168.0, "86 GNL_A: 0 MW, 87 GNL_B: -", 0.0],
        ]
        assert len(committed.notes) == 2

    def test_a_covered_stage_at_zero_mw_is_reported_as_zero_mw(
        self, synthetic_case_dir: Path, synthetic_source: SourceDeck, tmp_path: Path
    ) -> None:
        (committed,) = self._export_zero_mw(
            synthetic_case_dir, synthetic_source, tmp_path
        )

        assert committed.table is not None
        assert committed.table.columns == self._COLUMNS
        assert [row[3:] for row in committed.table.rows] == [
            ["86 GNL_A: 0 MW", 0.0]
        ] * 3

    def test_header_is_fixed_with_two_plants(
        self, gnl_case_dir: Path, gnl_source: SourceDeck, tmp_path: Path
    ) -> None:
        (committed,) = self._export_two_plants(gnl_case_dir, gnl_source, tmp_path)

        assert committed.table is not None
        assert committed.table.to_dict()["columns"] == self._COLUMNS

    def test_header_is_fixed_with_one_plant(
        self, synthetic_case_dir: Path, synthetic_source: SourceDeck, tmp_path: Path
    ) -> None:
        (committed,) = self._export_zero_mw(
            synthetic_case_dir, synthetic_source, tmp_path
        )

        assert committed.table is not None
        assert committed.table.to_dict()["columns"] == self._COLUMNS

    def test_a_case_without_gnl_thermals_emits_none(
        self, synthetic_case_dir: Path, synthetic_source: SourceDeck, tmp_path: Path
    ) -> None:
        _rewrite_gnl_lead(synthetic_case_dir, None)

        result, diagnostics = _run_export_on_policy(
            synthetic_case_dir,
            replace(synthetic_source, gnl_thermals=()),
            tmp_path,
            _make_pools_dict(),
            _make_manifest(),
        )

        assert result == 0
        assert _with_code(diagnostics, _COMMITTED) == []


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

    def test_cross_check_refuses_later_node_gnl(
        self, synthetic_case_dir: Path, synthetic_source: SourceDeck, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "dessem_fcf"
        with patch(
            "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
        ) as read:
            read.return_value = PolicyCheckpoint(_make_manifest(), _make_pools_dict())
            run_export(
                case_dir=synthetic_case_dir,
                source=synthetic_source,
                output_dir=output_dir,
                settings=ExportSettings(),
            )
        mapcut_path = output_dir / "mapcut.rv0"
        cortdeco_path = output_dir / "cortdeco.rv0"
        scalars = read_mapcut(mapcut_path).scalars
        cross_check_pair(mapcut_path, cortdeco_path)

        pi_gnl = 4  # three hydros, no travel-time plants
        later_node_record = next(
            index
            for index in range(scalars.numero_cortes)
            if node_and_iteration(index, scalars.numero_semanas)[0] > 0
        )
        with cortdeco_path.open("r+b") as handle:
            handle.seek(later_node_record * TAMANHO_CORTE + 4 + 8 * pi_gnl)
            handle.write(np.array([1.0], dtype="<f8").tobytes())

        with pytest.raises(LayoutError, match="first cut-building node"):
            cross_check_pair(mapcut_path, cortdeco_path)


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

    def test_failure_manifest_on_input_read_error(
        self,
        synthetic_case_dir: Path,
        synthetic_source: SourceDeck,
        tmp_path: Path,
    ) -> None:
        """A missing initial_conditions.json is a handled export failure."""
        (synthetic_case_dir / "initial_conditions.json").unlink()
        output_dir = tmp_path / "dessem_fcf"

        with (
            dx.collect() as diagnostics,
            patch(
                "novomodelo_bridge.dessem.exporters.fcf.pipeline.read_policy"
            ) as read,
        ):
            read.return_value = PolicyCheckpoint(_make_manifest(), _make_pools_dict())
            result = run_export(
                case_dir=synthetic_case_dir,
                source=synthetic_source,
                output_dir=output_dir,
                settings=ExportSettings(),
            )

        assert result == 1
        errors = [d for d in diagnostics if d.code == "dessem-fcf-export-failed"]
        assert len(errors) == 1
        assert "initial_conditions.json" in errors[0].summary
        assert errors[0].notes == ["Failed step: read settings-derived inputs"]

        manifest: dict[str, Any] = json.loads(
            (output_dir / "export_manifest.json").read_text()
        )
        assert manifest["status"] == "failed"
        assert manifest["failed_step"] == "read settings-derived inputs"
        assert not (output_dir / "mapcut.rv0").exists()
