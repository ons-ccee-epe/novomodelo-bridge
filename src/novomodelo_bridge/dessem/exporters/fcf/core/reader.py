"""Policy checkpoint reader over novomodelo-python's `novomodelo.results.load_policy`.

The checkpoint is materialized into frozen dataclasses at this boundary, and
every cut value leaves it in currency (R$): novomodelo stores intercepts and
coefficients in the run's scaled units and names the scale in each pool's
`cost_scale_factor`, so the reader multiplies by that factor. Dividing, or
passing the stored values through, ships a FCF off by the run's scale.

Coefficient vectors are sized by the pool's own entity manifest, never by
`state_dimension`, which is the study-global dimension.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_LOG = logging.getLogger(__name__)

# Entity type codes from novomodelo's policy.fbs schema.
VALID_ENTITY_TYPES = frozenset({0, 1, 2, 3})

# `YYYYMMDD` date fields carry i32::MIN on the families they do not apply to.
UNDATED = -2147483648

POLICY_OUTPUT_RELATIVE = Path("output")
POLICY_SUBDIR = "policy"


class PolicyFormatError(Exception):
    """Raised when the policy checkpoint is absent, unreadable, or inconsistent."""


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
    nodes: tuple[ManifestNodeRecord, ...]
    edges: tuple[ManifestEdgeRecord, ...]


@dataclass(frozen=True)
class EntitySlotRecord:
    """One state axis of a cut pool, in entity-manifest order.

    The dates are `YYYYMMDD` or `UNDATED`: `reference_date` is set on inflow
    lags, the half-open `[interval_start, interval_end)` on transit buckets and
    anticipated thermals (their delivery interval).
    """

    entity_type: int
    entity_id: int
    subindex: int
    was_active: bool
    reference_date: int
    interval_start: int
    interval_end: int


@dataclass(frozen=True, eq=False)
class AffinePieceRecord:
    """One affine piece of the value function, in R$.

    `eq=False` because `coefficients` is a numpy array, which would make a
    generated `__eq__` return an array rather than a bool.
    """

    slot_index: int
    iteration: int
    forward_pass_index: int
    intercept: float
    coefficients: NDArray[np.float64]
    is_active: bool


@dataclass(frozen=True)
class StageCutPool:
    """Every populated cut of one pool, plus the axes those cuts are expressed over.

    `stage_id` is the schema's pool id, the key `ManifestNodeRecord.pool_id`
    names. `graph_stage_id` is the actual stage key. `node_id` is `-1` for a
    shared pool owned by several nodes. `cost_scale_factor` is the scale the
    checkpoint stored the pieces in; `pieces` are already converted out of it.
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


@dataclass(frozen=True)
class PolicyCheckpoint:
    """The checkpoint's manifest and every pool it holds, keyed by pool id."""

    manifest: PolicyManifest
    pools: Mapping[int, StageCutPool]


def _manifest(metadata: Mapping[str, Any], n_pools_read: int) -> PolicyManifest:
    graph = metadata["graph_manifest"]
    return PolicyManifest(
        format_version=int(metadata["format_version"]),
        novomodelo_version=str(metadata["software_version"]),
        created_at=str(metadata["created_at"]),
        num_stages=int(metadata["num_stages"]),
        n_pools=int(graph.get("n_pools", n_pools_read)),
        completed_iterations=int(metadata["producer"]["completed_iterations"]),
        nodes=tuple(
            ManifestNodeRecord(
                id=int(node["id"]),
                stage_id=int(node["stage_id"]),
                pool_id=int(node["pool_id"]),
            )
            for node in graph["nodes"]
        ),
        edges=tuple(
            ManifestEdgeRecord(
                source_id=int(edge["source_id"]),
                target_id=int(edge["target_id"]),
                probability=float(edge["probability"]),
            )
            for edge in graph.get("edges", ())
        ),
    )


def _slot(entry: Mapping[str, Any], position: int, pool_id: int) -> EntitySlotRecord:
    slot = EntitySlotRecord(
        entity_type=int(entry["entity_type"]),
        entity_id=int(entry["entity_id"]),
        subindex=int(entry["subindex"]),
        was_active=bool(entry["was_active"]),
        reference_date=int(entry["reference_date"]),
        interval_start=int(entry["interval_start"]),
        interval_end=int(entry["interval_end"]),
    )
    if slot.entity_type not in VALID_ENTITY_TYPES:
        raise PolicyFormatError(
            f"unknown entity_type {slot.entity_type} at position {position} "
            f"of pool {pool_id}"
        )
    return slot


def _piece(
    cut: Mapping[str, Any], scale: float, n_slots: int, pool_id: int
) -> AffinePieceRecord:
    coefficients = np.asarray(cut["coefficients"], dtype=np.float64) * scale
    if coefficients.shape != (n_slots,):
        raise PolicyFormatError(
            f"pool {pool_id} cut at slot {cut['slot_index']} has "
            f"{coefficients.size} coefficients but its entity manifest has "
            f"{n_slots} slots"
        )
    coefficients.setflags(write=False)
    return AffinePieceRecord(
        slot_index=int(cut["slot_index"]),
        iteration=int(cut["iteration"]),
        forward_pass_index=int(cut["forward_pass_index"]),
        intercept=float(cut["intercept"]) * scale,
        coefficients=coefficients,
        is_active=bool(cut["is_active"]),
    )


def _pool(entry: Mapping[str, Any]) -> StageCutPool:
    pool_id = int(entry["stage_id"])
    scale = entry.get("cost_scale_factor")
    if scale is None or not math.isfinite(scale) or scale <= 0.0:
        raise PolicyFormatError(
            f"pool {pool_id} carries no usable cost_scale_factor ({scale!r}), "
            f"so its cut values cannot be converted to currency"
        )
    slots = tuple(
        _slot(slot, position, pool_id)
        for position, slot in enumerate(entry["entity_manifest"])
    )
    return StageCutPool(
        stage_id=pool_id,
        node_id=int(entry["node_id"]),
        graph_stage_id=int(entry["graph_stage_id"]),
        state_dimension=int(entry["state_dimension"]),
        capacity=int(entry["capacity"]),
        warm_start_count=int(entry["warm_start_count"]),
        populated_count=int(entry["populated_count"]),
        cost_scale_factor=float(scale),
        slots=slots,
        pieces=tuple(_piece(cut, scale, len(slots), pool_id) for cut in entry["cuts"]),
    )


def read_policy(case_dir: Path) -> PolicyCheckpoint:
    """Read `case_dir`'s `output/policy` checkpoint through novomodelo-python.

    Raises:
        PolicyFormatError: the checkpoint is missing, written by an
            incompatible novomodelo version, or internally inconsistent.
    """
    import novomodelo

    output_dir = case_dir / POLICY_OUTPUT_RELATIVE
    try:
        raw = novomodelo.results.load_policy(output_dir, policy_subdir=POLICY_SUBDIR)
    except OSError as exc:
        raise PolicyFormatError(
            f"cannot read the policy checkpoint under {output_dir / POLICY_SUBDIR}: "
            f"{exc}"
        ) from exc

    pools: dict[int, StageCutPool] = {}
    for entry in raw["stage_cuts"]:
        pool = _pool(entry)
        pools[pool.stage_id] = pool
    manifest = _manifest(raw["metadata"], len(pools))

    _LOG.info(
        "read policy checkpoint %s: format_version=%d novomodelo=%s num_stages=%d "
        "pools=%d nodes=%d edges=%d completed_iterations=%d",
        output_dir / POLICY_SUBDIR,
        manifest.format_version,
        manifest.novomodelo_version,
        manifest.num_stages,
        len(pools),
        len(manifest.nodes),
        len(manifest.edges),
        manifest.completed_iterations,
    )
    return PolicyCheckpoint(manifest=manifest, pools=pools)


def nodes_by_pool(manifest: PolicyManifest) -> dict[int, tuple[int, ...]]:
    """Group node ids by the pool that serves them, in ascending pool order."""
    grouped: dict[int, list[int]] = {}
    for node in manifest.nodes:
        grouped.setdefault(node.pool_id, []).append(node.id)
    return {pool_id: tuple(ids) for pool_id, ids in sorted(grouped.items())}


def assert_pools_declared(pool_ids: Sequence[int]) -> None:
    """Refuse a manifest that declares no cut pools at all."""
    if not pool_ids:
        raise PolicyFormatError(
            "the policy manifest declares no nodes, so the checkpoint carries "
            "no cut pools: there is no terminal pool to exclude and no trunk to convert"
        )


def select_pools(
    checkpoint: PolicyCheckpoint,
    pool_ids: Sequence[int],
    include_terminal: bool,
) -> dict[int, StageCutPool]:
    """The trunk pools, plus the terminal pool when `include_terminal`.

    `pool_ids` is the manifest's ascending pool order; its last id is the
    terminal pool.

    Raises:
        PolicyFormatError: the manifest names a pool the checkpoint lacks.
    """
    assert_pools_declared(pool_ids)
    wanted = list(pool_ids) if include_terminal else list(pool_ids[:-1])
    missing = [pool_id for pool_id in wanted if pool_id not in checkpoint.pools]
    if missing:
        raise PolicyFormatError(
            f"the policy manifest names pool(s) {missing} that the checkpoint "
            f"does not hold"
        )
    pools = {pool_id: checkpoint.pools[pool_id] for pool_id in wanted}
    _LOG.info(
        "selected %d pool(s): %s",
        len(pools),
        {pid: len(p.pieces) for pid, p in pools.items()},
    )
    return pools
