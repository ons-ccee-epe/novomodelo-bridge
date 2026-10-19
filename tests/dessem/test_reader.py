"""Tests for dessem/exporters/fcf/core/reader.py.

The round-trip tests author a tiny checkpoint with novomodelo-python's own writer,
so the reader is pinned against the format novomodelo actually emits.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from novomodelo_bridge.dessem.exporters.fcf.core.reader import (
    UNDATED,
    PolicyCheckpoint,
    PolicyFormatError,
    PolicyManifest,
    StageCutPool,
    nodes_by_pool,
    read_policy,
    select_pools,
)
from tests.conftest import requires_novomodelo_python, requires_writer_binding

_SCALE = 1000.0


def _slot(
    entity_type: int,
    entity_id: int,
    *,
    interval_start: int = UNDATED,
    interval_end: int = UNDATED,
) -> dict[str, object]:
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "subindex": 0,
        "was_active": True,
        "reference_date": UNDATED,
        "interval_start": interval_start,
        "interval_end": interval_end,
    }


_ENTITY_MANIFEST = [
    _slot(0, 0),
    _slot(0, 1),
    _slot(2, 0, interval_start=20260201, interval_end=20260301),
]


def _pool(pool_id: int, n_cuts: int, warm_start_count: int = 0) -> dict[str, object]:
    return {
        "stage_id": pool_id,
        "state_dimension": len(_ENTITY_MANIFEST),
        "capacity": 4,
        "warm_start_count": warm_start_count,
        "populated_count": n_cuts,
        "cost_scale_factor": _SCALE,
        "node_id": pool_id,
        "graph_stage_id": pool_id,
        "priced_state_date": UNDATED,
        "entity_manifest": _ENTITY_MANIFEST,
        "cuts": [
            {
                "slot_index": k,
                "iteration": k + 1,
                "forward_pass_index": 0,
                "intercept": 2.5 + k,
                "coefficients": [-1.0, -2.0, 0.5],
                "is_active": True,
                "scored": False,
            }
            for k in range(n_cuts)
        ],
    }


def _write_checkpoint(case_dir: Path) -> None:
    import novomodelo

    metadata = {
        "created_at": "2026-01-01T00:00:00Z",
        "num_stages": 3,
        "graph_manifest": {
            "n_pools": 3,
            "nodes": [{"id": i, "stage_id": i, "pool_id": i} for i in range(3)],
            "edges": [
                {"source_id": 0, "target_id": 1, "probability": 1.0},
                {"source_id": 1, "target_id": 2, "probability": 1.0},
            ],
        },
        "producer": {
            "completed_iterations": 2,
            "final_lower_bound": 0.0,
            "max_iterations": 2,
            "forward_passes": 1,
            "warm_start_cuts": 0,
            "rng_seed": 0,
            "cost_scale_factor": _SCALE,
        },
    }
    novomodelo.write_policy_checkpoint(
        case_dir / "output" / "policy",
        [_pool(0, 2), _pool(1, 2), _pool(2, 1)],
        metadata,
    )


@requires_novomodelo_python
@requires_writer_binding
def test_read_policy_converts_cut_values_to_currency(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    checkpoint = read_policy(tmp_path)

    piece = checkpoint.pools[0].pieces[1]
    assert piece.intercept == 3.5 * _SCALE
    assert piece.coefficients.tolist() == [-1.0 * _SCALE, -2.0 * _SCALE, 0.5 * _SCALE]
    assert not piece.coefficients.flags.writeable
    assert checkpoint.pools[0].cost_scale_factor == _SCALE


@requires_novomodelo_python
@requires_writer_binding
def test_read_policy_maps_manifest_graph_and_slot_dates(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    checkpoint = read_policy(tmp_path)

    manifest = checkpoint.manifest
    assert manifest.format_version == 4
    assert manifest.completed_iterations == 2
    assert [node.pool_id for node in manifest.nodes] == [0, 1, 2]
    assert [(e.source_id, e.target_id) for e in manifest.edges] == [(0, 1), (1, 2)]
    assert sorted(checkpoint.pools) == [0, 1, 2]
    gnl = checkpoint.pools[1].slots[2]
    assert (gnl.entity_type, gnl.interval_start, gnl.interval_end) == (
        2,
        20260201,
        20260301,
    )
    assert checkpoint.pools[1].slots[0].interval_start == UNDATED


@requires_novomodelo_python
def test_read_policy_without_a_checkpoint_raises_policy_format_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(PolicyFormatError, match="cannot read the policy checkpoint"):
        read_policy(tmp_path)


def _checkpoint(
    synthetic_manifest: PolicyManifest, pool: StageCutPool, pool_ids: list[int]
) -> PolicyCheckpoint:
    return PolicyCheckpoint(
        manifest=synthetic_manifest, pools={pool_id: pool for pool_id in pool_ids}
    )


def test_select_pools_excludes_the_terminal_pool_by_default(
    synthetic_manifest: PolicyManifest, synthetic_pool: StageCutPool
) -> None:
    checkpoint = _checkpoint(synthetic_manifest, synthetic_pool, [0, 1, 2])
    pool_ids = sorted(nodes_by_pool(synthetic_manifest))

    assert sorted(select_pools(checkpoint, pool_ids, include_terminal=False)) == [0, 1]
    assert sorted(select_pools(checkpoint, pool_ids, include_terminal=True)) == [
        0,
        1,
        2,
    ]


def test_select_pools_refuses_a_pool_the_checkpoint_lacks(
    synthetic_manifest: PolicyManifest, synthetic_pool: StageCutPool
) -> None:
    checkpoint = _checkpoint(synthetic_manifest, synthetic_pool, [0])

    with pytest.raises(PolicyFormatError, match=r"pool\(s\) \[1\]"):
        select_pools(checkpoint, [0, 1, 2], include_terminal=False)


def test_select_pools_refuses_a_manifest_without_pools(
    synthetic_manifest: PolicyManifest, synthetic_pool: StageCutPool
) -> None:
    checkpoint = _checkpoint(synthetic_manifest, synthetic_pool, [])

    with pytest.raises(PolicyFormatError, match="declares no nodes"):
        select_pools(checkpoint, [], include_terminal=False)
