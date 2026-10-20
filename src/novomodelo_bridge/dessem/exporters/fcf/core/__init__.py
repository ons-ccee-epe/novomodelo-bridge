"""Core mechanics for FCF export: readers, writers, validators."""

from novomodelo_bridge.dessem.exporters.fcf.core.premises import P3, P14, PREMISES
from novomodelo_bridge.dessem.exporters.fcf.core.reader import (
    UNDATED,
    AffinePieceRecord,
    EntitySlotRecord,
    ManifestEdgeRecord,
    ManifestNodeRecord,
    PolicyCheckpoint,
    PolicyFormatError,
    PolicyManifest,
    StageCutPool,
    assert_pools_declared,
    nodes_by_pool,
    read_policy,
    select_pools,
)

__all__ = [
    # Reader constants
    "UNDATED",
    # Reader exceptions
    "PolicyFormatError",
    # Reader dataclasses - manifest
    "PolicyCheckpoint",
    "PolicyManifest",
    "ManifestNodeRecord",
    "ManifestEdgeRecord",
    # Reader dataclasses - cuts
    "StageCutPool",
    "EntitySlotRecord",
    "AffinePieceRecord",
    # Reader functions
    "read_policy",
    "nodes_by_pool",
    "select_pools",
    "assert_pools_declared",
    # Premises
    "PREMISES",
    "P3",
    "P14",
]
