"""DESSEM FCF export: Novomodelo policy checkpoint → mapcut/cortdeco binary."""

from novomodelo_bridge.dessem.exporters.fcf.pipeline import EXPORT_PHASE_LABELS
from novomodelo_bridge.dessem.exporters.fcf.premises import P3, P14, PREMISES
from novomodelo_bridge.dessem.exporters.fcf.reader import (
    CUTS_SUBDIR,
    POLICY_MANIFEST_RELATIVE,
    SUPPORTED_FORMAT_VERSION,
    AffinePieceRecord,
    EntitySlotRecord,
    ManifestEdgeRecord,
    ManifestNodeRecord,
    PolicyFormatError,
    PolicyManifest,
    StageCutPool,
    assert_pools_declared,
    load_pools,
    nodes_by_pool,
    read_policy_manifest,
    read_stage_cuts,
)

__all__ = [
    # Pipeline
    "EXPORT_PHASE_LABELS",
    # Premises
    "PREMISES",
    "P3",
    "P14",
    # Reader constants
    "SUPPORTED_FORMAT_VERSION",
    "POLICY_MANIFEST_RELATIVE",
    "CUTS_SUBDIR",
    # Reader exceptions
    "PolicyFormatError",
    # Reader dataclasses - manifest
    "PolicyManifest",
    "ManifestNodeRecord",
    "ManifestEdgeRecord",
    # Reader dataclasses - cuts
    "StageCutPool",
    "EntitySlotRecord",
    "AffinePieceRecord",
    # Reader functions
    "read_policy_manifest",
    "read_stage_cuts",
    "nodes_by_pool",
    "load_pools",
    "assert_pools_declared",
]
