"""Policy checkpoint reader: reads Novomodelo FlatBuffers policy files.

Both readers materialize their tables into frozen dataclasses at this boundary;
no live FlatBuffers table travels deeper into the package, because a table is a
view over a buffer whose lifetime the caller cannot see.

Coefficient vectors are sized by the pool's own entity manifest, never by
`StageCuts.state_dimension`, which is the study-global dimension.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from Novomodelo.IO.Policy.CheckpointManifest import CheckpointManifest
from Novomodelo.IO.Policy.StageCuts import StageCuts

_LOG = logging.getLogger(__name__)

SUPPORTED_FORMAT_VERSION = 1
_MIN_BUFFER_BYTES = 8

# FlatBuffers surfaces a malformed buffer as whatever the underlying struct or
# index operation raises. Enumerated rather than caught blanket.
_DECODE_ERRORS = (
    IndexError,
    OverflowError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    struct.error,
)

# Entity type codes from Novomodelo's policy.fbs schema.
VALID_ENTITY_TYPES = frozenset({0, 1, 2, 3})

# Standard checkpoint paths relative to case directory.
POLICY_MANIFEST_RELATIVE = Path("output/policy/manifest.bin")
CUTS_SUBDIR = Path("output/policy/cuts")


class PolicyFormatError(Exception):
    """Raised when a policy file is absent, truncated, or not the supported format."""


# ---------------------------------------------------------------------------
# Manifest dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestNodeRecord:
    """One node of the policy graph."""

    id: int
    stage_id: int
    pool_id: int


@dataclass(frozen=True)
class ManifestEdgeRecord:
    """One transition of the policy graph, with its probability."""

    source_id: int
    target_id: int
    probability: float


@dataclass(frozen=True)
class PolicyManifest:
    """Study-level checkpoint metadata and the policy graph."""

    format_version: int
    novomodelo_version: str
    created_at: str
    num_stages: int
    n_pools: int
    completed_iterations: int
    cost_scale_factor: float
    nodes: tuple[ManifestNodeRecord, ...]
    edges: tuple[ManifestEdgeRecord, ...]


# ---------------------------------------------------------------------------
# Cut pool dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntitySlotRecord:
    """One state axis of a cut pool, in entity-manifest order."""

    entity_type: int
    entity_id: int
    subindex: int
    was_active: bool
    delivery_date: int


@dataclass(frozen=True, eq=False)
class AffinePieceRecord:
    """One affine piece of the value function.

    `eq=False` because `coefficients` is a numpy array, which would make a
    generated `__eq__` return an array rather than a bool.
    """

    piece_id: int
    slot_index: int
    iteration: int
    forward_pass_index: int
    intercept: float
    coefficients: NDArray[np.float64]
    is_active: bool


@dataclass(frozen=True)
class StageCutPool:
    """Every cut of one pool, plus the axes those cuts are expressed over.

    `stage_id` is the schema's pool id, the `cuts/<pool>.bin` key.
    `graph_stage_id` is the actual stage key.
    `node_id` is `-1` for a shared pool owned by several nodes.
    """

    stage_id: int
    node_id: int
    graph_stage_id: int
    state_dimension: int
    capacity: int
    warm_start_count: int
    populated_count: int
    cost_scale_factor: float
    slots: tuple[EntitySlotRecord, ...]
    pieces: tuple[AffinePieceRecord, ...]
    active_cut_indices: tuple[int, ...]


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------


def _read_buffer(path: Path) -> bytes:
    """Read raw bytes from a file, validating minimum size."""
    if not path.is_file():
        raise PolicyFormatError(f"policy file not found: {path}")
    raw = path.read_bytes()
    if len(raw) < _MIN_BUFFER_BYTES:
        raise PolicyFormatError(
            f"policy file too short: {path} holds {len(raw)} bytes, "
            f"minimum is {_MIN_BUFFER_BYTES}"
        )
    return raw


def _decode_text(value: bytes | None) -> str:
    """Decode a FlatBuffers string field."""
    return value.decode("utf-8") if value is not None else ""


def _read_coefficients(piece: Any, length: int) -> NDArray[np.float64]:
    """Extract coefficient array from a FlatBuffers piece."""
    if length == 0:
        return np.empty(0, dtype=np.float64)
    array: NDArray[np.float64] = np.asarray(
        piece.CoefficientsAsNumpy(), dtype=np.float64
    )
    array.setflags(write=False)
    return array


# ---------------------------------------------------------------------------
# Manifest reader
# ---------------------------------------------------------------------------


def read_policy_manifest(path: Path) -> PolicyManifest:
    """Read `manifest.bin`, rejecting unsupported format versions."""
    buffer = _read_buffer(path)
    try:
        table = CheckpointManifest.GetRootAs(buffer, 0)
        format_version = table.FormatVersion()
    except _DECODE_ERRORS as exc:
        raise PolicyFormatError(f"cannot decode policy manifest {path}: {exc}") from exc

    if format_version != SUPPORTED_FORMAT_VERSION:
        raise PolicyFormatError(
            f"unsupported policy format_version {format_version} in {path}; "
            f"this tool supports version {SUPPORTED_FORMAT_VERSION}"
        )

    try:
        nodes = tuple(
            ManifestNodeRecord(
                id=node.Id(), stage_id=node.StageId(), pool_id=node.PoolId()
            )
            for node in (table.Nodes(i) for i in range(table.NodesLength()))
        )
        edges = tuple(
            ManifestEdgeRecord(
                source_id=edge.SourceId(),
                target_id=edge.TargetId(),
                probability=edge.Probability(),
            )
            for edge in (table.Edges(i) for i in range(table.EdgesLength()))
        )
        manifest = PolicyManifest(
            format_version=format_version,
            novomodelo_version=_decode_text(table.NovomodeloVersion()),
            created_at=_decode_text(table.CreatedAt()),
            num_stages=table.NumStages(),
            n_pools=table.NPools(),
            completed_iterations=table.CompletedIterations(),
            cost_scale_factor=table.CostScaleFactor(),
            nodes=nodes,
            edges=edges,
        )
    except _DECODE_ERRORS as exc:
        raise PolicyFormatError(f"cannot decode policy manifest {path}: {exc}") from exc

    _LOG.info(
        "read policy manifest %s: format_version=%d num_stages=%d n_pools=%d "
        "nodes=%d edges=%d completed_iterations=%d",
        path,
        manifest.format_version,
        manifest.num_stages,
        manifest.n_pools,
        len(manifest.nodes),
        len(manifest.edges),
        manifest.completed_iterations,
    )
    return manifest


def nodes_by_pool(manifest: PolicyManifest) -> dict[int, tuple[int, ...]]:
    """Group node ids by the pool that serves them, in ascending pool order."""
    grouped: dict[int, list[int]] = {}
    for node in manifest.nodes:
        grouped.setdefault(node.pool_id, []).append(node.id)
    return {pool_id: tuple(ids) for pool_id, ids in sorted(grouped.items())}


# ---------------------------------------------------------------------------
# Stage cuts reader
# ---------------------------------------------------------------------------


def read_stage_cuts(path: Path) -> StageCutPool:
    """Read one `cuts/<pool>.bin`, sizing coefficients by the entity manifest."""
    buffer = _read_buffer(path)
    try:
        table = StageCuts.GetRootAs(buffer, 0)
        manifest_length = table.EntityManifestLength()

        slots = tuple(
            EntitySlotRecord(
                entity_type=slot.EntityType(),
                entity_id=slot.EntityId(),
                subindex=slot.Subindex(),
                was_active=bool(slot.WasActive()),
                delivery_date=slot.DeliveryDate(),
            )
            for slot in (table.EntityManifest(i) for i in range(manifest_length))
        )

        for position, slot in enumerate(slots):
            if slot.entity_type not in VALID_ENTITY_TYPES:
                raise PolicyFormatError(
                    f"unknown entity_type {slot.entity_type} at position "
                    f"{position} in {path}"
                )

        pieces: list[AffinePieceRecord] = []
        for index in range(table.CutsLength()):
            piece = table.Cuts(index)
            coef_len = piece.CoefficientsLength()
            if coef_len != manifest_length:
                raise PolicyFormatError(
                    f"coefficient/manifest mismatch in {path}: piece {index} has "
                    f"{coef_len} coefficients but manifest has {manifest_length}"
                )
            pieces.append(
                AffinePieceRecord(
                    piece_id=piece.PieceId(),
                    slot_index=piece.SlotIndex(),
                    iteration=piece.Iteration(),
                    forward_pass_index=piece.ForwardPassIndex(),
                    intercept=piece.Intercept(),
                    coefficients=_read_coefficients(piece, coef_len),
                    is_active=bool(piece.IsActive()),
                )
            )

        pool = StageCutPool(
            stage_id=table.StageId(),
            node_id=table.NodeId(),
            graph_stage_id=table.GraphStageId(),
            state_dimension=table.StateDimension(),
            capacity=table.Capacity(),
            warm_start_count=table.WarmStartCount(),
            populated_count=table.PopulatedCount(),
            cost_scale_factor=table.CostScaleFactor(),
            slots=slots,
            pieces=tuple(pieces),
            active_cut_indices=tuple(
                int(table.ActiveCutIndices(i))
                for i in range(table.ActiveCutIndicesLength())
            ),
        )
    except _DECODE_ERRORS as exc:
        raise PolicyFormatError(f"cannot decode stage cuts {path}: {exc}") from exc

    _LOG.info(
        "read stage cuts %s: stage_id=%d node_id=%d slots=%d pieces=%d",
        path,
        pool.stage_id,
        pool.node_id,
        len(pool.slots),
        len(pool.pieces),
    )
    return pool


# ---------------------------------------------------------------------------
# Pool loading utilities
# ---------------------------------------------------------------------------


def _cuts_path(case_dir: Path, pool_id: int) -> Path:
    """Return the path to a pool's cuts file."""
    return case_dir / CUTS_SUBDIR / f"{pool_id:03d}.bin"


def assert_pools_declared(pool_ids: Sequence[int]) -> None:
    """Refuse a manifest that declares no cut pools at all."""
    if not pool_ids:
        raise PolicyFormatError(
            "the policy manifest declares no nodes, so the checkpoint carries "
            "no cut pools: there is no terminal pool to exclude and no trunk to convert"
        )


def load_pools(
    case_dir: Path,
    pool_ids: Sequence[int],
    include_terminal: bool,
) -> dict[int, StageCutPool]:
    """Load trunk pools, optionally including terminal for ECO CSV.

    Args:
        case_dir: Novomodelo case directory
        pool_ids: Sorted pool IDs from manifest (ascending)
        include_terminal: If True, also load the terminal pool (last in pool_ids)

    Returns:
        Dictionary mapping pool_id to StageCutPool
    """
    assert_pools_declared(pool_ids)
    terminal_pool_id = pool_ids[-1]
    trunk_ids = pool_ids[:-1]

    pools: dict[int, StageCutPool] = {
        pool_id: read_stage_cuts(_cuts_path(case_dir, pool_id)) for pool_id in trunk_ids
    }

    if include_terminal:
        pools[terminal_pool_id] = read_stage_cuts(
            _cuts_path(case_dir, terminal_pool_id)
        )
        _LOG.info(
            "loaded terminal pool %d with %d pieces",
            terminal_pool_id,
            pools[terminal_pool_id].populated_count,
        )

    _LOG.info(
        "loaded %d pool(s): %s",
        len(pools),
        {pid: len(p.pieces) for pid, p in pools.items()},
    )
    return pools
