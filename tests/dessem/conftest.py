"""Pytest fixtures for dessem/exporters/fcf tests.

Provides synthetic Novomodelo case inputs and policy checkpoint fixtures.
All fixtures are generated programmatically - no external data dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import (
    BlockInfo,
    BusInfo,
    CaseInputs,
    HydroInfo,
    StageInfo,
    ThermalInfo,
)
from novomodelo_bridge.dessem.exporters.fcf.core.layout import CutBlockOffsets
from novomodelo_bridge.dessem.exporters.fcf.core.mapper import (
    SourceGnlThermal,
    SourceHydro,
)
from novomodelo_bridge.dessem.exporters.fcf.core.reader import (
    AffinePieceRecord,
    EntitySlotRecord,
    ManifestEdgeRecord,
    ManifestNodeRecord,
    PolicyManifest,
    StageCutPool,
)
from novomodelo_bridge.dessem.exporters.fcf.source import SourceDeck


@pytest.fixture
def synthetic_manifest() -> PolicyManifest:
    """A minimal synthetic policy manifest with 3 stages, 4 nodes, 3 pools."""
    nodes = tuple(
        ManifestNodeRecord(id=i, stage_id=min(i, 2), pool_id=min(i, 2))
        for i in range(4)
    )
    edges = (
        ManifestEdgeRecord(source_id=0, target_id=1, probability=0.5),
        ManifestEdgeRecord(source_id=0, target_id=2, probability=0.5),
        ManifestEdgeRecord(source_id=1, target_id=3, probability=1.0),
    )
    return PolicyManifest(
        format_version=4,
        novomodelo_version="0.1.0",
        created_at="2026-01-01T00:00:00Z",
        num_stages=3,
        n_pools=3,
        completed_iterations=10,
        nodes=nodes,
        edges=edges,
    )


@pytest.fixture
def synthetic_slots() -> tuple[EntitySlotRecord, ...]:
    """Synthetic entity slots: 3 hydro storage + 2 anticipated thermal."""
    sentinel = -2147483648
    return (
        EntitySlotRecord(
            entity_type=0,
            entity_id=1,
            subindex=0,
            was_active=True,
            reference_date=sentinel,
            interval_start=sentinel,
            interval_end=sentinel,
        ),
        EntitySlotRecord(
            entity_type=0,
            entity_id=2,
            subindex=0,
            was_active=True,
            reference_date=sentinel,
            interval_start=sentinel,
            interval_end=sentinel,
        ),
        EntitySlotRecord(
            entity_type=0,
            entity_id=3,
            subindex=0,
            was_active=True,
            reference_date=sentinel,
            interval_start=sentinel,
            interval_end=sentinel,
        ),
        EntitySlotRecord(
            entity_type=2,
            entity_id=100,
            subindex=0,
            was_active=True,
            reference_date=sentinel,
            interval_start=20260201,
            interval_end=20260301,
        ),
        EntitySlotRecord(
            entity_type=2,
            entity_id=100,
            subindex=1,
            was_active=True,
            reference_date=sentinel,
            interval_start=20260301,
            interval_end=20260401,
        ),
    )


@pytest.fixture
def synthetic_pieces(
    synthetic_slots: tuple[EntitySlotRecord, ...],
) -> tuple[AffinePieceRecord, ...]:
    """Synthetic affine pieces (cuts) for testing."""
    n_slots = len(synthetic_slots)
    pieces = []
    for i in range(5):
        coefficients = np.array(
            [100.0 * (i + 1) + j for j in range(n_slots)], dtype=np.float64
        )
        coefficients.setflags(write=False)
        pieces.append(
            AffinePieceRecord(
                slot_index=i,
                iteration=i // 2,
                forward_pass_index=i % 2,
                intercept=1000.0 * (i + 1),
                coefficients=coefficients,
                is_active=True,
            )
        )
    return tuple(pieces)


@pytest.fixture
def synthetic_pool(
    synthetic_slots: tuple[EntitySlotRecord, ...],
    synthetic_pieces: tuple[AffinePieceRecord, ...],
) -> StageCutPool:
    """A synthetic stage cut pool."""
    return StageCutPool(
        stage_id=0,
        node_id=0,
        graph_stage_id=0,
        state_dimension=len(synthetic_slots),
        capacity=100,
        warm_start_count=0,
        populated_count=len(synthetic_pieces),
        cost_scale_factor=1.0,
        slots=synthetic_slots,
        pieces=synthetic_pieces,
    )


@pytest.fixture
def synthetic_inputs() -> CaseInputs:
    """Synthetic Novomodelo case inputs."""
    stages = tuple(
        StageInfo(
            id=i,
            start_date=f"2026-01-0{i + 1}",
            end_date=f"2026-01-0{i + 8}",
            blocks=(
                BlockInfo(id=0, name="leve", hours=24.0),
                BlockInfo(id=1, name="media", hours=72.0),
                BlockInfo(id=2, name="pesada", hours=72.0),
            ),
            storage_state=True,
            inflow_lags_state=False,
        )
        for i in range(3)
    )
    hydros = (
        HydroInfo(
            id=1,
            name="UHE_A",
            downstream_id=None,
            min_storage_hm3=0.0,
            max_storage_hm3=100.0,
        ),
        HydroInfo(
            id=2,
            name="UHE_B",
            downstream_id=1,
            min_storage_hm3=0.0,
            max_storage_hm3=200.0,
        ),
        HydroInfo(
            id=3,
            name="UHE_C",
            downstream_id=1,
            min_storage_hm3=0.0,
            max_storage_hm3=150.0,
        ),
    )
    thermals = (
        ThermalInfo(id=100, name="GNL_A", bus_id=0, lead_time_hours=1512.0),
        ThermalInfo(id=101, name="CONV_A", bus_id=1, lead_time_hours=None),
    )
    buses = (
        BusInfo(id=0, name="SE"),
        BusInfo(id=1, name="S"),
        BusInfo(id=2, name="NE"),
        BusInfo(id=3, name="N"),
        BusInfo(id=4, name="FC"),
        BusInfo(id=5, name="IV"),
    )
    return CaseInputs(
        stages=stages,
        hydros=hydros,
        thermals=thermals,
        buses=buses,
        annual_discount_rate=0.12,
    )


@pytest.fixture
def synthetic_offsets() -> CutBlockOffsets:
    """Offsets for 3 hydros, 2 GNL submarkets, 3 stages, 3 blocks."""
    return CutBlockOffsets(
        rhs=0,
        pi_varm=1,
        pi_gnl=4,  # 1 + 3 hydros
        ncoef=4 + 2 * 3 * 3,  # pi_varm + n_sbm_gnl * n_estagios * n_patamares
    )


@pytest.fixture
def synthetic_source(tmp_path: Path) -> SourceDeck:
    """The deck `synthetic_case_dir` was converted from; codes differ from ids."""
    return SourceDeck(
        directory=tmp_path / "deck",
        revision="rv0",
        hydros={
            1: SourceHydro(code=101, name="UHE_A"),
            2: SourceHydro(code=102, name="UHE_B"),
            3: SourceHydro(code=103, name="UHE_C"),
        },
        gnl_thermals=(SourceGnlThermal(code=86, name="GNL_A", lag_months=3),),
    )


@pytest.fixture
def synthetic_case_dir(tmp_path: Path, synthetic_inputs: CaseInputs) -> Path:
    """Create a minimal synthetic Novomodelo case directory with JSON files."""
    case_dir = tmp_path / "synthetic_case"
    case_dir.mkdir()
    system_dir = case_dir / "system"
    system_dir.mkdir()

    # stages.json
    stages_data = {
        "policy_graph": {"annual_discount_rate": synthetic_inputs.annual_discount_rate},
        "stages": [
            {
                "id": s.id,
                "start_date": s.start_date,
                "end_date": s.end_date,
                "blocks": [
                    {"id": b.id, "name": b.name, "hours": b.hours} for b in s.blocks
                ],
                "state_variables": {
                    "storage": s.storage_state,
                    "inflow_lags": s.inflow_lags_state,
                },
            }
            for s in synthetic_inputs.stages
        ],
    }
    (case_dir / "stages.json").write_text(json.dumps(stages_data, indent=2))

    # hydros.json
    hydros_data = {
        "hydros": [
            {
                "id": h.id,
                "name": h.name,
                "downstream_id": h.downstream_id,
                "reservoir": {
                    "min_storage_hm3": h.min_storage_hm3,
                    "max_storage_hm3": h.max_storage_hm3,
                },
            }
            for h in synthetic_inputs.hydros
        ]
    }
    (system_dir / "hydros.json").write_text(json.dumps(hydros_data, indent=2))

    # thermals.json
    thermals_data = {
        "thermals": [
            {
                "id": t.id,
                "name": t.name,
                "bus_id": t.bus_id,
                **(
                    {"anticipated_config": {"lead_time_hours": t.lead_time_hours}}
                    if t.lead_time_hours is not None
                    else {}
                ),
            }
            for t in synthetic_inputs.thermals
        ]
    }
    (system_dir / "thermals.json").write_text(json.dumps(thermals_data, indent=2))

    # buses.json
    buses_data = {
        "buses": [{"id": b.id, "name": b.name} for b in synthetic_inputs.buses]
    }
    (system_dir / "buses.json").write_text(json.dumps(buses_data, indent=2))

    return case_dir
